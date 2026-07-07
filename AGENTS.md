# AGENTS.md

This file provides guidance to **AI Agents** (Antigravity, Cursor, Claude Code, Windsurf, etc.) and terminal tools (like warp.dev) when working with code in this repository.

## Project Overview

**vailá** (Versatile Anarcho Integrated Liberation Ánalysis) is an open-source Python 3.12 multimodal toolbox for biomechanical data analysis. It integrates IMU, motion capture, markerless tracking (MediaPipe, YOLO), force plates, EMG, GNSS/GPS, and other sensor data through a Tkinter-based GUI. Licensed under AGPLv3.

## Build & Run Commands

### Hybrid CPU laptop vs NVIDIA workstation

The repo ships **several `pyproject_*.toml` templates**. The checked-in **`pyproject.toml` matches `pyproject_universal_cpu.toml`**: portable **CPU** PyTorch (laptops / no CUDA). That manifest defines optional extras `dev`, `upscaler`, `sam`, and **`fifa`** (FIFA Skeletal Tracking Light pipeline: vendored `sam_3d_body` + PyTorch Lightning stack) — it does **not** define `gpu` (so `uv sync --extra gpu` fails until you switch templates).

**Recommended for any dev (Linux / macOS / WSL / Windows):** use the unified interactive bootstrap. It auto-detects OS + NVIDIA, suggests the right template + extras, then runs `uv lock` + `uv sync`:

```bash
# Linux / macOS / WSL / Git Bash
bash bin/setup_pyproject.sh                           # interactive, auto-detect
bash bin/setup_pyproject.sh --target=linux-cuda --extras=gpu,sam,sapiens --yes
bash bin/setup_pyproject.sh --target=cpu --non-interactive

# Windows PowerShell
pwsh bin/setup_pyproject.ps1                          # interactive, auto-detect
pwsh bin/setup_pyproject.ps1 -Target win-cuda -Extras gpu,sam -Yes
```

Flags: `--target=auto|cpu|linux-cuda|win-cuda|macos`, `--extras=a,b,c`, `--non-interactive`, `--yes`, `--no-lock`, `--no-sync`, `--help`. Use `--non-interactive --no-sync` in CI to just swap the template + lock without installing.

**Legacy per-platform switchers** (kept as thin wrappers around `setup_pyproject.sh/.ps1` for backward compatibility):

| Platform | Switch (from repo root) | Then |
|----------|-------------------------|------|
| Linux CUDA 12.8 | `bash bin/use_pyproject_linux_cuda.sh` | `uv sync --extra gpu` and optionally `--extra sam` |
| Windows CUDA 12.1 | `pwsh bin/use_pyproject_win_cuda.ps1` | same |
| macOS (Metal) | `bash bin/use_pyproject_macos_metal.sh` | `uv sync` |

**Back to portable CPU** (e.g. same clone on a laptop): `bash bin/use_pyproject_universal_cpu.sh` (Linux/macOS) or `pwsh bin/use_pyproject_universal_cpu.ps1` (Windows), then `uv sync`.

Each switch runs `uv lock` and rewrites `uv.lock` for that hardware matrix. The default lock in git targets **CPU**; CUDA users regenerate locally after switching.

SAM 3 video (`vaila_sam.py`) requires **NVIDIA CUDA** at runtime (`torch.cuda.is_available()`), even if the `sam` extra is installed. There is **no** CPU-only or **macOS Metal/MPS** path in this integration; `--frame-by-frame` only lowers **VRAM on CUDA**, not a CPU fallback. Without CUDA, use other vailá modules (e.g. Markerless 2D / YOLO) or run on a CUDA workstation or cloud GPU. Checkpoint auto-detection supports both `vaila/models/sam3/` and repo-root `models/sam3/`.

**Sapiens2 Pose (optional):** `uv sync --extra sapiens` plus `bash bin/setup_sapiens2.sh` (clones into `.local/third_party/sapiens2/`, editable install, downloads `facebook/sapiens2-pose-1b` + `facebook/detr-resnet-101-dc5` into `vaila/models/sapiens2/`). GUI: Frame B → **YOLO + FB** → **Sapiens2 Pose** (`vaila/vaila_sapiens.py`). Default model **1B** fits RTX 4090 24 GiB. Help: `vaila/help/vaila_sapiens.md`. License: Meta Sapiens2 License (not AGPL).

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
├── bin/setup_sapiens2.sh/.ps1    ← Clones sapiens2 + downloads pose + DETR weights
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
- **GUI→CLI mirror:** any module with a CLI must print copy-paste equivalent commands on GUI **Run** using the `>>` prefix (absl logging eats `[bracketed]` stdout). Chooser **YOLO + FB** prints launcher CLI per button; full args in `vaila_sam`, `vaila_sapiens`, `yolov26track track`, `yolotrain`. Reference: `docs/vaila_buttons/yolo-fb.md`, `yolotrain._format_training_cli_command`, `getpixelvideo` `>> Equivalent CLI`.

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

### Save-freeze fix + verbose README + sam_bbox_tracks alias (June 2026, v0.3.55)

After yesterday's smart-loader work users could **load** SAM3 `sam_tracks.csv`
into `getpixelvideo.py` and convert bboxes → markers, but clicking **Save**
froze the pygame GUI for 10+ minutes on long broadcast clips, forcing
`sudo kill -9`. Three issues fixed today:

1. **Wrong Save branch.** After the smart loader converted bboxes to markers,
   the Save handler still hit `elif csv_loaded and tracking_data:` and called
   `export_labeling_dataset`, which extracts every annotated frame to disk
   (`cv2.VideoCapture` seek + decode per frame + 3 file writes per frame).
   For a 16 693-frame clip with 248 K bboxes this is tens of minutes of
   blocking I/O on the main pygame thread. **Fix:** new state flag
   `bbox_converted_to_markers` (set in `load_tracking_csv` after a successful
   anchor conversion) routes Save through `save_coordinates`
   (regular `*_markers.csv`) instead. Users who *want* the YOLO dataset
   export simply answer the anchor prompt with Enter (keep bboxes as overlay
   only); Save then takes the dataset path.

2. **`save_coordinates` itself was O(N×slots) pandas `.at[]` calls.** On the
   same 248 K-bbox case this was minutes of `df.at[frame, "pN_x"] = …`. Now
   vectorised with NumPy bulk assignment into a `(n_frames, max_points*2)`
   `float64` array, then wrapped in a single `pd.DataFrame`. **Benchmark on a
   16 693 × 62 grid with 248 310 entries: 0.56 s** (vs. effectively hung).

3. **GUI never told the user a save was in flight.** New
   `_flush_save_message(screen, text)` helper paints a yellow banner directly
   to the pygame surface and flips the display **before** any long save
   begins (both the dataset export path and the marker save path). No more
   "is it frozen?" panic.

**Discoverability + provenance** (same release):

- New shared helpers in `vaila/vaila_sam.py`: `_write_sam_run_readme()` and
  `_make_sam_bbox_tracks_alias()` (plus `SAM_OUTPUT_FILE_GLOSSARY` constant).
  Both the chunked merge path (`_merge_chunk_outputs` caller) and the
  single-pass writer (`run_sam3_on_video`) now go through them.
- Every SAM3 run now writes a **verbose `README_sam.txt`** — explicit schema,
  units and downstream role for every produced file (`sam_tracks.csv`,
  `sam_frames_meta.csv`, `sam_points.csv`, `sam_id_map.csv`,
  `sam_contours.json`, `sam_masks_manifest.csv`, `<video>_sam_overlay.mp4`,
  `masks/`, `FAILED_sam.txt`). The previous README was 8 lines and only
  named the chunk stats.
- A sibling **`sam_bbox_tracks.csv`** is created as a POSIX hardlink
  (zero disk cost, same inode) or copy fallback. Discoverable name with
  `bbox` in it; consumers that still read `sam_tracks.csv` are unaffected.
  Smart loader detects formats by columns, not filename, so both names just
  work.

Version sync: `0.3.55 / 15 June 2026` on `vaila.py`, `vaila/vaila_sam.py`,
`vaila/sam_postprocess.py`, `vaila/getpixelvideo.py`,
`vaila/help/vaila_sam.{md,html}`, `vaila/help/getpixelvideo.{md,html}`.
Tests: 338 passed, 1 skipped, 1 deselected (the unrelated `tugturn`
Qt-platform `xcb` env failure).

Skill: `.claude/skills/getpixelvideo-tracking-loader/SKILL.md` § *Save
behaviour (v0.3.55)*. SAM3 helper section:
`.claude/skills/sam3-video/SKILL.md` § *Output Format*.

**Follow-up later same day:** the ML dataset writers
(`export_labeling_dataset`, `export_pose_dataset`,
`_export_all_labels_view`) are *legitimately* slow — they extract +
re-encode thousands of video frames + write 3 files per frame. Without
terminal output users still assumed the GUI hung mid-save. Added three
top-level helpers in `vaila/getpixelvideo.py`: `_save_banner(title, detail)`
(boxed banner with destination + counts before each save begins),
`_save_done(message)` (completion tail), `_try_import_tqdm()` (soft import,
`tqdm` is already transitive via ultralytics/pytorch). Wired into all three
writers with a `tqdm` bar per train/val/test split. **Note for future
agents:** absl logging (installed by mediapipe / opencv on import) eats
`[bracketed]` prefixes from stdout, so the banner uses `>>
vaila/getpixelvideo:` instead. tqdm writes to stderr and is unaffected.
Documented in `docs/sessions/2026-06-15-getpixel-savefreeze-readme-bbox-alias.md`
§ 8.

### Cross-Chunk Tracklet Linking + getpixelvideo smart loader (June 2026, v0.3.54)

Two related items shipped together on `reidtrain`. Full transcript:
`docs/sessions/2026-06-14-getpixel-sam3-crosschunk.md`.

**1. `vaila/vaila_sam.py` — Cross-Chunk Tracklet Linking.**

Symptom: when the coordinator fell back to `_process_video_chunked` on long
1080p clips (see *Coordinator pattern + chunked fallback* above), each chunk
allocated its own local object IDs starting at 1. The merged
`sam_tracks.csv` therefore had random IDs across chunk boundaries, breaking
any downstream Re-ID / trajectory analysis.

Root cause: `_build_cross_chunk_id_maps` called `_assignment_min_cost`, but
that helper was never defined in `vaila_sam.py` (it was assumed importable
from `reid_markers.py`). Result: `NameError` at runtime — the entire stitch
silently no-op'd.

Fix: defined `_assignment_min_cost` inline (SciPy Hungarian +
greedy fallback). Wired the full 5-step pipeline that the user specified:
sliding-window overlap of **2 frames** between adjacent chunks,
per-local-id feature cache, bipartite IoU + centroid-distance cost matrix,
Hungarian assignment, linked-list merging with `min_iou ≥ 0.05` and
`max_centroid_dist_px ≤ 180` gates. Tunable defaults at
`_build_cross_chunk_id_maps(max_centroid_dist_px=180.0, min_iou=0.05)`; no
CLI flags yet. Test:
`tests/test_vaila_sam.py::test_merge_chunk_outputs` (assertions updated
for the new 0-indexed global IDs).

**2. `vaila/getpixelvideo.py` — Intelligent Load Tracking CSV.**

Replaced the brittle “must have a `Frame` column” loader with
auto-detection of 5 formats: `sam_tracks`, `sam_frames_meta` (normalised
bbox, converted to pixel using video w/h), `sam_points`, `yolo_multi`
(`all_id_detection.csv`), `yolo_single` (`person_id_NN.csv`). Unknown
files still fall back to the legacy YOLO parser.

Added bbox → marker **anchor prompt** via `show_input_dialog`
(`1=center 2=bottom 3=top 4=left 5=right`, Enter = keep overlay only).
Helpers: `_BBOX_ANCHOR_ALIASES`, `_anchor_xy_from_bbox`,
`_detect_frame_col`, `_detect_tracking_format`, `_iter_bboxes_from_df`,
`bboxes_to_marker_coordinates`. Once an anchor is chosen the bboxes become
regular editable / saveable vailá markers. Skill:
`.claude/skills/getpixelvideo-tracking-loader/SKILL.md`.

Version sync (`0.3.54` / 14 June 2026) applied to `vaila.py`,
`vaila/vaila_sam.py`, `vaila/sam_postprocess.py`, `vaila/getpixelvideo.py`,
`vaila/help/vaila_sam.{md,html}`, `vaila/help/index.{md,html}`, and
`README.md`. Tests pass except for the unrelated `tests/test_tugturn_integration.py::test_cli_end_to_end` Qt-plugin environment failure (already documented in *Sports field CLI and GUI integration fixes*).

### File Manager Tkinter fixes and hybrid SSH Transfer (July 2026, session 2026-07-01)

**Module:** `vaila/filemanager.py` and `vaila.py`.

Symptom 1: Clicking the **Transfer** button on the main GUI printed debug messages and then immediately closed/crashed the entire Python/Tkinter GUI process with exit code 139 (Segmentation Fault).
Symptom 2: Executing `vaila/transfer.sh` via the CLI failed with `rsync: [sender] change_dir "/mnt/disco2tb1/Downloads" failed: No such file or directory` when the user typed or copy-pasted trailing spaces.
Symptom 3: Buttons like **Copy**, **Move**, and **Import** caused UI freezes or Tcl/Tk crashes on Linux X11 due to creating duplicate `tk.Tk()` root instances and starting secondary event loops (`root.mainloop()`).

Root cause 1: `_transfer_file_gui()` returned immediately without blocking. When the local variables (specifically the `StringVar` variables bound to entry fields) went out of scope, Python garbage-collected them. Their `__del__` destructor unregistered the variables from the Tcl interpreter. When the Tk event loop tried to render or process events for the widgets bound to those deleted variables, it dereferenced a NULL pointer and segfaulted.
Root cause 2: Interactive user input in the shell script kept trailing whitespace (e.g. `/mnt/disco2tb1/Downloads         `), causing `rsync` to search for a non-existent path.
Root cause 3: Python's Tkinter wrapper does not support multiple `tk.Tk()` root window loops running simultaneously in the same process.

Fixes:
- **Tcl/Tk Segmentation Fault Fix**: Bound the `StringVar` instances to the window object as attributes, and blocked returning from `_transfer_file_gui()` using `root.wait_window(transfer_window)` when run in embedded mode to keep the local scope alive.
- **CLI Trailing Spaces Fix**: Added automatic leading/trailing whitespace trimming to all user inputs in `vaila/transfer.sh` using `xargs`.
- **Duplicate Root Window Fixes**: Refactored `copy_file()`, `move_file()`, and `import_file()` to use a hybrid window management pattern: they detect the existing `tk._default_root`, spawn a transient modal `tk.Toplevel` dialog, and wait using `root.wait_window()`. They only fall back to `tk.Tk()` and `mainloop()` when run standalone outside the main GUI.
- **Hybrid GUI-Terminal Transfer**: Because `rsync` requires a real interactive terminal (TTY) for SSH password input, the Transfer button was rewritten to collect parameters in a transient modal GUI dialog, write them to a temp script, and launch the script in a new terminal emulator window (`gnome-terminal`, etc.) so the user can safely type their password.

Version sync: `0.3.67 / 01 July 2026` on `vaila.py`, `vaila/filemanager.py`, `vaila/transfer.sh`, `vaila/help/filemanager.{md,html}`, `vaila/help/index.{md,html}`, and `README.md`.
Tests: 430 passed, 1 skipped.
Skill: `.claude/skills/filemanager-tkinter-and-ssh-transfer/SKILL.md`.

### Unified Geometric Re-ID module + full plan implementation (July 2026, v0.3.68)

**New module:** `vaila/geometric_reid.py` — shared Hungarian matching, IoU
helpers, velocity-direction cost, mask IoU, homography gate, and
`GeometricFrameLinker` class. Eliminates 3× duplicate `_assignment_min_cost`
and 2× near-identical greedy linker implementations.

**Modules refactored:**
- `vaila/yolov26track.py` — `_GeometricTrackLinker` is now an alias for
  `GeometricFrameLinker`; greedy replaced by Hungarian; CLI gets
  `--reid-max-gap`, `--reid-max-dist`, `--reid-min-iou`,
  `--reid-direction-weight`, `--reid-homography`, `--appearance-reid`;
  BoT-SORT custom YAML with `with_reid + GMC` in CLI via
  `_build_botsort_custom_yaml`; `--no-pose` path calls
  `_apply_geometric_stabilize_to_buffer` + writes `yolo_reid_links.csv`;
  GUI: stabilize checkbox moved to Run Mode; ReID tuning fields + appearance
  ReID checkbox added.
- `vaila/vaila_sam.py` — `_stabilize_sam_track_ids` uses
  `GeometricFrameLinker` (Hungarian + velocity); `_build_cross_chunk_id_maps`
  imports `assignment_min_cost` + `bbox_iou_xywh` from shared module; new
  `mask_iou_weight` cost term when mask PNGs exist; `--overlap-frames N`
  CLI flag (default 2).
- `vaila/reid_markers.py` — imports `assignment_min_cost` from
  `geometric_reid`; local copy removed; new
  `geometric_reid_align_markers_bidirectional()` (forward + backward merge).
- `vaila/reid_yolotrack.py` — parser fixed for `person_id_01.csv` (zero-padded);
  dead `StrongSORT` / `pip install` removed; new headless
  `run_appearance_reid_on_tracking_dir()` for yolov26track hook; output CSVs
  use `_id_{NN:02d}` naming.

**Tests added:**
- `tests/test_geometric_reid.py` — Hungarian crossing, velocity penalty, IoU
- `tests/test_reid_yolotrack.py` — CSV filename parsing
- Extended `tests/test_yolov26track_pose_reid.py` — stabilize buffer + links CSV

Version sync: `0.3.68 / 04 July 2026`. Tests: 15 passed (Re-ID subset),
full SAM chunk overlap test passes.

### Default post-processing + VAILA anchor CSVs (July 2026, v0.3.69)

**Changed defaults:**
- `--postprocess-points` CLI default: `none` → `all`
- GUI `Post-process points` combobox default: `none` → `all`

`sam_points.csv` + `sam_id_map.csv` are now generated automatically after
every SAM batch or single-video run. No need to pass `--postprocess-points`
explicitly unless you want `none` to skip.

**New outputs:** five simple VAILA-style anchor CSVs per video:
`sam_vaila_center.csv`, `sam_vaila_bottom.csv`, `sam_vaila_top.csv`,
`sam_vaila_left.csv`, `sam_vaila_right.csv`. Format: `frame,x1,y1,...,xN,yN`
(one x,y pair per tracked object). Each file uses a different bbox anchor
(center, foot, head, left, right). Ready for direct loading in
`getpixelvideo` / `rec2d`.

**Modules changed:** `vaila/sam_postprocess.py` (new `write_vaila_anchor_csvs`,
`write_vaila_anchor_csvs_for_batch`, `VAILA_ANCHORS`), `vaila/vaila_sam.py`
(all 4 postprocess call sites wired, glossary updated, GUI button relabelled).

Tests: `tests/test_sam_postprocess.py` — 26 passed (10 new anchor tests).

Version sync: `0.3.69 / 04 July 2026`.

### YOLO + FB chooser + Sapiens2 Pose (July 2026, v0.3.71)

**GUI (`vaila.py`):**
- Frame B button **YOLO + SAM** renamed to **YOLO + FB**
- Chooser adds **Sapiens2 Pose** → `vaila/vaila_sapiens.py` (308 kp, CUDA)
- Bootstrap: `bash bin/setup_sapiens2.sh`; extra `uv sync --extra sapiens`

**New module:** `vaila/vaila_sapiens.py` — DETR + Sapiens2 ViT pose; default model `1b` for RTX 4090.

Tests: `tests/test_vaila_sapiens.py`. Help: `vaila/help/vaila_sapiens.md`.

### GUI→CLI mirror for YOLO + FB stack (July 2026, v0.3.72)

**Convention:** modules with CLI print copy-paste commands on GUI **Run** using `>>` prefix
(absl eats `[bracketed]` stdout). Chooser prints launcher CLI per button.

**Code:**
- `vaila.py` — `_print_yolo_fb_launch()` on all chooser buttons
- `vaila_sapiens.py` — `_format_sapiens_cli_command` + print on Run
- `vaila_sam.py` — `_build_sam_cli_argv` + `_print_sam_equivalent_cli`
- `yolov26track.py` — `_format_track_cli_command` (one `track` per video); pose workflow hints
- `yolotrain.py` — launch line + `_format_training_cli_command` in training thread

**Docs:** `docs/vaila_buttons/yolo-fb.md`, helps rebranded to **YOLO + FB** path.

**Skill:** `.claude/skills/yolo-fb-gui-cli/SKILL.md`
**Session log:** `docs/sessions/2026-07-06-yolo-fb-gui-cli-mirror.md`

Version sync: `0.3.72 / 06 July 2026`.

### Sapiens2 duplicate output directory fix (July 2026, v0.3.76)

**Module:** `vaila/vaila_sapiens.py`.

**Symptom:** CLI and GUI runs left two `processed_sapiens_<timestamp>/` folders — one empty, one with CSVs/overlay.

**Root cause:** Default subprocess-per-video isolation spawned workers that called `main()` without `--output-base`. Each child minted a **new** timestamp directory before handling `--video-output-dir`, while the parent batch already owned the real output tree.

**Fix:**
- `main()` returns early when `--video-output-dir` is set — no second timestamp folder.
- `_build_isolated_sapiens_cmd()` passes `--output-base` from the parent batch to each isolated worker.

**Tests:** `tests/test_vaila_sapiens.py::test_build_isolated_sapiens_cmd_passes_output_base`.

**Docs:** `vaila/help/vaila_sapiens.{md,html}` § Output directory (v0.3.76); session `docs/sessions/2026-07-07-sapiens-output-dir-fix.md`.

Version sync: `0.3.76 / 07 July 2026`.

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
