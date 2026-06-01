# AGENTS.md

This file provides guidance to **AI Agents** (Antigravity, Cursor, Claude Code, Windsurf, etc.) and terminal tools (like warp.dev) when working with code in this repository.

## Project Overview

**vailá** (Versatile Anarcho Integrated Liberation Ánalysis) is an open-source Python 3.12 multimodal toolbox for biomechanical data analysis. It integrates IMU, motion capture, markerless tracking (MediaPipe, YOLO), force plates, EMG, GNSS/GPS, and other sensor data through a Tkinter-based GUI. Licensed under AGPLv3.

## Build & Run Commands

### Hybrid CPU laptop vs NVIDIA workstation

The repo ships **several `pyproject_*.toml` templates**. The checked-in **`pyproject.toml` matches `pyproject_universal_cpu.toml`**: portable **CPU** PyTorch (laptops / no CUDA). That manifest defines optional extras `dev`, `upscaler`, `sam`, and **`fifa`** (FIFA Skeletal Tracking Light pipeline: vendored `sam_3d_body` + PyTorch Lightning stack) — it does **not** define `gpu` (so `uv sync --extra gpu` fails until you switch templates).

**Workstation with NVIDIA CUDA** — copy the platform template, regenerate the lock, then sync:

| Platform | Switch (from repo root) | Then |
|----------|-------------------------|------|
| Linux CUDA 12.8 | `bash bin/use_pyproject_linux_cuda.sh` | `uv sync --extra gpu` and optionally `--extra sam` |
| Windows CUDA 12.1 | `pwsh bin/use_pyproject_win_cuda.ps1` | same |
| macOS (Metal) | `bash bin/use_pyproject_macos_metal.sh` | `uv sync` |

**Back to portable CPU** (e.g. same clone on a laptop): `bash bin/use_pyproject_universal_cpu.sh` (Linux/macOS) or `pwsh bin/use_pyproject_universal_cpu.ps1` (Windows), then `uv sync`.

Each switch runs `uv lock` and rewrites `uv.lock` for that hardware matrix. The default lock in git targets **CPU**; CUDA users regenerate locally after switching.

SAM 3 video (`vaila_sam.py`) requires **NVIDIA CUDA** at runtime (`torch.cuda.is_available()`), even if the `sam` extra is installed. There is **no** CPU-only or **macOS Metal/MPS** path in this integration; `--frame-by-frame` only lowers **VRAM on CUDA**, not a CPU fallback. Without CUDA, use other vailá modules (e.g. Markerless 2D / YOLO) or run on a CUDA workstation or cloud GPU. Checkpoint auto-detection supports both `vaila/models/sam3/` and repo-root `models/sam3/`.

**FIFA Skeletal Tracking Light (optional):** `uv sync --extra fifa` (workstation: combine with CUDA template + `--extra gpu`). `sam_3d_body/` is **not committed** — clone it with `bash bin/setup_fifa_sam3d.sh` (or `pwsh bin/setup_fifa_sam3d.ps1` on Windows), which also downloads the gated `facebook/sam-3d-body-dinov3` weights into `vaila/models/sam-3d-dinov3/`. Vendored MIT starter-kit utilities live in `vaila/fifa_starter_lib/` (`camera_tracker.py`, `postprocess.py`, `pitch_points.txt`; see `vaila/fifa_starter_lib/VENDOR.md`). CLI: `uv run vaila/vaila_sam.py fifa <subcommand> --help` with subcommands `bootstrap` (symlinks + sequences + pitch_points), `prepare`, `boxes`, `preprocess`, `baseline`, **`dlt-export`** (FIFA `cameras/*.npz` → per-frame `.dlt2d`/`.dlt3d` via `vaila/fifa_to_dlt.py` for **`rec2d.py` / `rec3d.py`** on moving broadcast cameras), `pack`. Use **`rec2d_one_dlt2d.py` / `rec3d_one_dlt3d.py` only for fixed cameras** (single DLT row). Companion tool `vaila/soccerfield_calib.py` (button **Soccer-Field Calib** in Frame C of `vaila.py`) fits a **single-frame** DLT2D homography from 29 FIFA keypoints; GUI **FIFA cams→DLT** exports per-frame DLT after `baseline --export-camera`. Tests: `uv run pytest tests/test_fifa_skeletal_pipeline.py tests/test_fifa_bootstrap.py tests/test_fifa_to_dlt.py tests/test_soccerfield_calib.py -v`. Full `data/` layout (`cameras/`, `boxes/`, …) still comes from the official starter kit / Hugging Face dataset when available.

```bash
# Run the application (recommended)
uv run vaila.py

# Install dependencies (after choosing the right pyproject.toml as above)
uv sync                          # default / universal CPU template
uv sync --extra sam              # optional SAM 3 deps (HF gated weights; CUDA at runtime)
uv sync --extra gpu              # only after Linux/Windows CUDA template is active
uv sync --extra gpu --extra sam  # CUDA template + SAM
uv sync --extra fifa             # FIFA skeletal pipeline (SAM 3D Body + Lightning; use with GPU template for CUDA)

# Lint and format
uv run ruff check vaila/           # Lint
uv run ruff check vaila/ --fix     # Lint with auto-fix
uv run ruff format vaila/          # Format

# Type checking
uv run ty check vaila/

# Run a single module standalone (some modules support CLI)
uv run vaila/interp_smooth_split.py -i /path/to/csv_dir -c smooth_config.toml

# Run automated tests
uv run pytest tests/                           # Run all tests
uv run pytest tests/test_vaila_and_jump.py -v   # Run jump specific tests
uv run pytest tests/test_tugturn.py -v          # Run TUG specific tests
uv run pytest tests/test_dlt_rec.py -v          # Run DLT/Rec math tests
uv run pytest tests/test_dlt_rec_integration.py -v # Run DLT/Rec pipeline tests
uv run pytest tests/test_vaila_sam.py -v           # SAM helpers + GUI Help smoke; GPU: tests/SAM/README.md
uv run pytest tests/test_fifa_skeletal_pipeline.py -v  # FIFA layout/packaging unit tests (no GPU)
uv run pytest tests/test_fifa_bootstrap.py -v          # FIFA data-layout bootstrap (symlinks + sequences)
uv run pytest tests/test_soccerfield_calib.py -v       # Soccer-field DLT2D homography tests
uv run pytest tests/test_fifa_to_dlt.py -v             # FIFA cameras NPZ -> per-frame DLT2D/DLT3D

# Install git hooks (pre-commit blocks files ≥20 MiB)
bash install-hooks.sh
```

The project uses `pytest` for automated testing.

- `tests/test_vaila_and_jump.py` — Unit tests for biomechanical calculations.
- `tests/test_vaila_and_jump_integration.py` — Integration tests for full analysis pipelines using sample data.
- The `tests/vaila_and_jump/` directory contains the sample data (CSV, TOML) used by these tests.

**Milestone (02 March 2026):** Refactored `vaila_and_jump.py` (v0.1.3), `vaila/tugturn.py`, and the DLT/Reconstruction suite (`dlt2d.py`, `dlt3d.py`, `rec2d_one_dlt2d.py`, `rec3d.py`, `rec3d_one_dlt3d.py`). Fixed all Ruff/Ty lint and type errors, added CLI/headless support, and established a comprehensive automated test suite across `tests/`.

## Mandatory: Update metadata on any script change

When you change **any** Python script (`*.py`) anywhere in this repo, also update user-facing metadata so version/date stay consistent.

### Checklist

- **Script header**: in the edited `*.py`, update top module docstring/header fields:
  - **Update Date**: today
  - **Version**: **global vailá version** (same as `vaila.py` header/banner)
- **Main entry point**: if change impacts GUI/CLI banner, also update `vaila.py`:
  - header **Update Date** + **Version**
  - any printed/banner strings that embed version/date
- **Installers**: review/update if install/run UX impacted:
  - `install_vaila_linux.sh`, `install_vaila_mac.sh`, `install_vaila_win.ps1`, `install-hooks.sh`
- **README**: update root `README.md` line `Last updated: YYYY-MM-DD` to today
- **Help docs**: keep help in sync with edited script:
  - main index `vaila/help/index.md` + `vaila/help/index.html` (“Generated on”)
  - edited module help `vaila/help/<module>.md` + `vaila/help/<module>.html` (Version + Updated)

## External unified pitch dataset (YOLO retrain, outside the repo)

The merged **32 pitch keypoint** tree from `vaila.fifa_dataset_builder` is designed to live on disk **outside** the git clone (large image banks). Ultralytics training points at `<dataset_root>/unified/data.yaml` with an **absolute** `data=` path. After QA on a flat `check_all_labels/` export, use `vaila.fifa_check_labels_dedupe` and `vaila.fifa_dataset_train_readiness` (`--prune-unified-to-flat`) so `unified/` matches human-validated samples. Narrative: **`docs/fifa_workflow.md` §4.5**; GUI companion help: **`vaila/help/soccerfield_keypoints_ai.md`** (Training → Option B).

```bash
uv run pytest tests/test_fifa_dataset_builder.py tests/test_fifa_check_labels_dedupe.py \
  tests/test_fifa_dataset_train_readiness.py -v
```

## Repo structure

```
vaila/                 ← root
├── vaila.py           ← Main Tkinter GUI entry point
├── vaila/             ← All analysis modules (package)
│   ├── fifa_starter_lib/  ← Vendored MIT starter-kit utils (camera_tracker, postprocess, pitch_points)
│   ├── fifa_bootstrap.py  ← `fifa bootstrap` helper (symlinks + sequences + pitch_points)
│   └── soccerfield_calib.py  ← Companion DLT2D calibration (29 FIFA keypoints)
├── bin/setup_fifa_sam3d.sh/.ps1  ← Clones sam_3d_body + downloads gated weights
├── sam_3d_body/       ← Cloned locally by the setup script (NOT committed)
├── tests/             ← pytest test suite
├── docs/              ← Documentation
├── .claude/
│   ├── agents/        ← Specialized agent roles (biomechanics, GUI, video, tests)
│   ├── skills/        ← Step-by-step skills (new module, port MATLAB)
│   └── commands/      ← Slash-command specs (/check, /new-module)
├── .cursor/rules/     ← Cursor IDE rules
├── pyproject.toml     ← Default (CPU)
├── pyproject_*.toml   ← Platform-specific templates
└── uv.lock
```

**`vaila/models/`:** Reference **`.csv`** (and similar small files) are **tracked**. Downloaded weights (**`.pt`**, **`.ckpt`**, **`.onnx`**, **`.engine`**, **`.task`**, **`.safetensors`**, etc.) and **`vaila/models/**/.cache/`** are **gitignored**; fetch via first run or **`hf download`** (e.g. [facebook/sam3](https://huggingface.co/facebook/sam3), `facebook/sam-3d-body-dinov3`). Small default **`.pkl`** (walkway ML) may stay tracked if **< 20 MiB**. Pre-commit blocks staged files **≥ 20 MiB**. Details: [CONTRIBUTING.md](CONTRIBUTING.md#vaila-models-directory). **`tests/SAM/*.mp4`** is gitignored (place sample locally; see `tests/SAM/README.md`).

## Platform-Specific Configuration

The project uses a **template-based pyproject.toml system** for hardware-specific dependencies. Before creating a venv, the correct template must be copied to `pyproject.toml`:

- `pyproject_win_cuda12.toml` — Windows NVIDIA CUDA 12.1
- `pyproject_linux_cuda12.toml` — Linux NVIDIA CUDA 12.8
- `pyproject_macos.toml` — macOS Metal/MPS (Apple Silicon)
- `pyproject_universal_cpu.toml` — CPU-only fallback

The install scripts (`install_vaila_linux.sh`, `install_vaila_mac.sh`, `install_vaila_win.ps1`) handle this automatically. For manual setup, copy the template **before** running `uv python pin` / `uv venv`.

## Architecture

### Entry Point & GUI (`vaila.py`)

`vaila.py` is the main entry point. It defines the `Vaila(tk.Tk)` class which builds the entire GUI. The interface is organized into three frames:

- **Frame A (File Manager):** Rename, import, export, copy, move, remove, tree, find, transfer
- **Frame B (Multimodal Analysis):** IMU, MoCap, Markerless 2D/3D, EMG, Force Plate, GNSS, etc.
- **Frame C (Tools):** Data Files (CSV editing, C3D conversion, DLT reconstruction), Video/Image processing, Visualization

Each GUI button dispatches to a function in `vaila/` using **lazy imports** — modules are imported inside handler methods to avoid loading the entire dependency graph at startup. Two dispatch patterns are used:

1. **Direct import + call:** `from vaila import module; module.run_function()` — runs in the same process
2. **Subprocess launch via `run_vaila_module()`:** Launches `python -m vaila.module_name` in a separate process — used when Tkinter conflicts could occur (e.g., modules that create their own Tk root)

### Package Structure (`vaila/`)

The `vaila/` package contains ~100 self-contained analysis modules. Each module typically:

- Has a `run_*()` or `analyze_*()` entry function called from the GUI
- Uses Tkinter `filedialog` to prompt users for input directories/files
- Reads CSV/C3D data using `pandas`/`numpy`/`ezc3d`
- Processes data and writes results (CSV + PNG plots) to timestamped output subdirectories

Key shared modules:

- `data_processing.py` — CSV/C3D reading with auto-header detection
- `filtering.py` / `filter_utils.py` — Butterworth and FIR filter implementations
- `common_utils.py` — Header detection and data reshaping for CSV files
- `dialogsuser.py` / `dialogsuser_cluster.py` — Reusable Tkinter input dialogs for sample rate, file type
- `filemanager.py` — All file management operations (rename, copy, move, transfer via SSH)
- `hardware_manager.py` — GPU/CPU detection, TensorRT auto-export for YOLO models
- `interp_smooth_split.py` — Interpolation, smoothing, and splitting pipeline (supports both GUI and CLI modes); configured via `smooth_config.toml`

### Import Conventions

Modules use **relative imports** when imported as part of the package (`from .readcsv import ...`) with a `try/except` fallback to absolute imports for standalone execution (`from readcsv import ...`). This dual-import pattern is common throughout the codebase.

### Ruff Configuration

Ruff is configured in `pyproject.toml` with:

- Target: Python 3.12, line length 100
- Enabled rule sets: E, W, F, I, N, NPY, UP, B, C4, SIM
- Ignored: `E501` (line length handled by formatter), `N806`/`N803` (scientific code uses uppercase variable names like X, Y, F)
- `__init__.py` files ignore `F401` (unused imports are intentional re-exports)

## Conventions

- **Python version:** 3.12 (strictly `>=3.12,<3.13`)
- **GUI framework:** Tkinter (standard library) — do not introduce other GUI frameworks
- **Scientific computing:** Use `numpy` and `pandas` efficiently for data processing
- **Naming:** Scientific code may use uppercase variable names (X, Y, Z, F, etc.) — this is acceptable and suppressed in linting
- **Output pattern:** Analysis modules write results to timestamped subdirectories (e.g., `processed_linear_lowess_YYYYMMDD_HHMMSS/`)
- **Build system:** `hatchling` backend, managed via `uv`

## History (cross-IDE memory)

Use this as a quick lookup for known-hard issues that already have a documented fix (full debugging session, hypotheses, runtime evidence and fix details live in the matching skill under `.claude/skills/`).

### SAM3 batch CUDA OOM cascade (April 2026, debug session 42b4a5)

Symptom: `vaila/vaila_sam.py` batch over a directory failed with `CUDA OOM while loading the video into SAM3` on most videos after the first OOM, even with the 256→64→32 retry ladder, and even after `_release_sam3_gpu_memory()` (gc + empty_cache).

Root cause: **two compounding leaks** on the OOM path of `predictor.handle_request("start_session")`:

1. The live `Exception e` traceback retained ~7 GiB of inner-frame SAM3 tensors. Calling `_release_sam3_gpu_memory()` *inside* the `except` block was a no-op because `e` was still alive — the release only worked at the **top of the next iteration** (after `e` was implicitly deleted).
2. ~13 GiB of orphan tensors held in **SAM3's C++ workspace pools** (CUDA-side state opaque to Python's gc) survived `predictor = None`, `gc.collect()` and `torch.cuda.empty_cache()`. They could **only be released by killing the Python process**.

Fix in `vaila/vaila_sam.py`:

- Moved `_release_sam3_gpu_memory()` to the start of each retry-loop iteration in `_process_one_video_with_oom_retry`.
- Added **subprocess-per-video isolation** in the CLI batch loop (`main()`): for each video, spawn `python vaila_sam.py --input <single.mp4> --video-output-dir <out>` so OS-level process death guarantees a clean GPU for the next video. Default ON when batch has >1 video; disable with `--no-isolate-batch` only for debugging. New internal flag `--video-output-dir` lets the child write directly to the parent-supplied dir without creating its own `processed_sam_TS/` wrapper.
- Added **rich outputs** for downstream multi-camera / 3D reconstruction: overlay can draw bbox/ID/score/contours and the run exports `sam_contours.json` (polygons), `sam_tracks.csv` (long bbox+area stats), and `sam_masks_manifest.csv` (mask index).

Full hypothesis log, runtime evidence, and code map: see `.claude/skills/sam3-video/SKILL.md` § *Why subprocess-per-video* and § *Cascading OOM in batch*.

### Sports field CLI and GUI integration fixes (April 2026, session efe5a0)

Symptom 1: `vaila/drawsportsfields.py` crashed with `qt.qpa.plugin` errors when run in CLI mode (`--field`) on Linux, while working fine when launched from the main `vaila.py` GUI.
Symptom 2: `App` class initialization in `vaila/markerless_3d_analysis.py` failed with `ArgumentError` (expected `Tk`, found `Toplevel`) when integrated into the notebook interface.
Symptom 3: Matplotlib reported `Module matplotlib.cm has no member rainbow` due to deprecated attribute access.

Root cause 1: Matplotlib defaulted to the `QtAgg` backend in standalone CLI mode. On systems with missing or broken Qt dependencies, this caused an immediate crash. Additionally, the `--field` flag was incorrectly triggering the full Tkinter GUI loop instead of a simple plot.
Root cause 2: The `App` class was defined inside an `if __name__ == "__main__":` block and its constructor type-hint was restricted to `tk.Tk`, preventing its use as a `tk.Toplevel` child window.
Root cause 3: Modern Matplotlib versions require `plt.get_cmap("name")` instead of `plt.cm.name`.

Fixes:
- Added `matplotlib.use("TkAgg")` before `pyplot` imports in `drawsportsfields.py` to ensure cross-platform compatibility without Qt.
- Refactored `drawsportsfields.py` CLI logic to use static `plt.show()` instead of the full `run_soccerfield()` GUI when arguments are passed.
- Moved `App` class out of `__main__` in `markerless_3d_analysis.py` and updated its type hint to `tk.Tk | tk.Toplevel`.
- Added native FIFA layout support via `soccerfield_ref3d_fifa.csv` and `SPORT_REGISTRY["fifa"]`.
- Updated all colormap accesses to use `plt.get_cmap()`.

Full details: see `.claude/skills/sports-field-visualization/SKILL.md`.

### Soccer field AI keypoints (YOLO pose + video CSV) (April 2026, session 254a97)

**Module:** `vaila/soccerfield_keypoints_ai.py` — Ultralytics local weights or Roboflow API; video mode writes `field_keypoints_video.csv`, `field_keypoints_getpixelvideo.csv`, `field_keypoints_overlay_markers.csv`, optional `field_keypoints_overlay.mp4`.

**Training dataset (local YOLO):** `vaila/models/hf_datasets/football-pitch-detection/data/data.yaml` (`kpt_shape: [32, 3]`). Prefer `imgsz=1280`, `mosaic=0`, `erasing=0`, and moderate `pose`/`kobj` weights; naive `mosaic=1` + small `imgsz` can collapse all keypoints into a tiny cluster while box mAP stays high.

**Handoff for other IDEs / agents:** read `.claude/skills/soccer-field-keypoints-yolo/SKILL.md` for exact train/export/infer commands, weight path resolution (Ultralytics run dir suffix `-N`), and CSV semantics.


### Crop Face GUI integration and Help opener (June 2026, session 2026-06-01)

**Module:** `vaila/crop_faces_atletas.py` — athlete face photo cropper by
Abel Gonçalves Chinaglia. It now follows the project GUI pattern: select input
photo directory first, then output directory. It is wired to **Frame C -> Video
and Image -> C_B_r1_c2 - Crop Face**. Help files live at
`vaila/help/crop_faces_atletas.md` and `.html`; `README.md` and
`vaila/help/index.*` list the new button.

**Model path:** the MediaPipe detector is downloaded on first use into the
Git-ignored `vaila/models/crop_face/face_detector.task`; provision explicitly
with `uv run python vaila/crop_faces_atletas.py --download-model`. Legacy
locations `vaila/crop_face/models/face_detector.task`,
`vaila/models/face_detector.task`, and repo-root `models/face_detector.task`
remain auto-detected. If download fails, the GUI asks the user to select a
compatible `.task` or `.tflite` file. Keep downloaded model files out of git.

**Help opener fix:** the main `Help` button in `vaila.py` must use
`webbrowser.open_new_tab(Path(...).as_uri())` for `vaila/help/index.html`.
Do not use shell `open`/`xdg-open` through `os.system` for this button; on Linux
that can open an IDE/editor instead of the browser depending on user file
associations.

## Caveman mode (optional)

[Caveman](https://github.com/JuliusBrussee/caveman) is a skills/plugin pack for AI coding agents (Claude Code, Cursor, Gemini CLI, Windsurf, Copilot, and 30+ others). It steers the model toward terse replies: fewer filler words and articles, typically **~65–75% fewer output tokens** while keeping technical content intact.

### Install

The upstream installer auto-detects supported agents. **Node.js + `npx`** must be on your PATH (the installer uses `npx skills add`).

- **macOS / Linux / WSL / Git Bash**

```bash
curl -fsSL https://raw.githubusercontent.com/JuliusBrussee/caveman/main/install.sh | bash
```

- **Windows (PowerShell)**

```powershell
irm https://raw.githubusercontent.com/JuliusBrussee/caveman/main/install.ps1 | iex
```

Re-run the same command to refresh. Optional: `--with-init` (see upstream README) to write per-repo rule files into the current directory.

### Enable / disable in a session

- **On:** `/caveman`, or phrases like “talk like caveman” / “less tokens please”. On Codex: `$caveman`.
- **Off:** `stop caveman` or `normal mode`.

**Intensity:** `/caveman lite` (trim fluff, keep grammar), `/caveman full` (default: minimal articles/sentences), `/caveman ultra` (telegraphic), `/caveman wenyan` (upstream style).

**Auto-clarity:** drop caveman tone for security warnings, irreversible actions, or when the user is confused; resume after.

**Boundaries:** generated code and formal commit/PR prose can stay normal; the style target is conversational assistant output.

### Extra upstream tools (input + workflow)

- **`/caveman:compress <file>`** — rewrite large project memory files (e.g. `CLAUDE.md`) in the same terse style to cut **input** tokens on every read (upstream cites ~46% average on those files).
- **`/caveman-commit`** — very short commit messages (focus on *why*).
- **`/caveman-review`** — one-line PR review comments.
- **caveman-shrink (MCP)** — optional proxy that compresses MCP tool/resource descriptions to save system-side tokens.

### Cursor in this repo

Repo-local Cursor rule (always-on terse baseline for the IDE): `.cursor/rules/caveman.mdc`. Upstream skills install is separate; use the `curl` / `irm` commands above.

### In-chat rules (mirror)

Respond terse like smart caveman. All technical substance stay. Only fluff die.

- Drop: articles (a/an/the), filler (just/really/basically), pleasantries, hedging
- Fragments OK. Short synonyms. Technical terms exact. Code unchanged
- Pattern: [thing] [action] [reason]. [next step].
- Not: “Sure! I'd be happy to help you with that.”
- Yes: “Bug in auth middleware. Fix:”

## Security

This is an **open-source (AGPL-3.0)** repository. Do **not** commit API keys, tokens, or credential files. See **[SECURITY.md](SECURITY.md)** and **[CONTRIBUTING.md](CONTRIBUTING.md)**.
