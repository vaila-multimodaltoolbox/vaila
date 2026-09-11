# AGENTS.md

Guidance for **AI Agents** (Antigravity, Cursor, Claude Code, Windsurf, etc.) and terminal tools (warp.dev) working with code in this repo.

## Project Overview

**vailá** (Versatile Anarcho Integrated Liberation Ánalysis) — open-source Python 3.12 multimodal toolbox, biomechanical data analysis. Integrates IMU, motion capture, markerless tracking (MediaPipe, YOLO), force plates, EMG, GNSS/GPS, other sensor data via Tkinter GUI. Licensed AGPLv3.

## Build & Run Commands

### Hybrid CPU laptop vs NVIDIA workstation

Repo ships **several `pyproject_*.toml` templates**. Checked-in **`pyproject.toml` matches `pyproject_universal_cpu.toml`**: portable **CPU** PyTorch (laptops / no CUDA). That manifest defines optional extras `dev`, `upscaler`, `sam`, **`fifa`** (FIFA Skeletal Tracking Light pipeline: vendored `sam_3d_body` + PyTorch Lightning stack) — does **not** define `gpu` (so `uv sync --extra gpu` fails till template switch).

**Recommended, any dev (Linux / macOS / WSL / Windows):** unified interactive bootstrap. Auto-detects OS + NVIDIA, suggests right template + extras, runs `uv lock` + `uv sync`:

```bash
# Linux / macOS / WSL / Git Bash
bash bin/setup_pyproject.sh                           # interactive, auto-detect
bash bin/setup_pyproject.sh --target=linux-cuda --extras=gpu,sam,sapiens --yes
bash bin/setup_pyproject.sh --target=cpu --non-interactive

# Windows PowerShell
pwsh bin/setup_pyproject.ps1                          # interactive, auto-detect
pwsh bin/setup_pyproject.ps1 -Target win-cuda -Extras gpu,sam -Yes
```

Flags: `--target=auto|cpu|linux-cuda|win-cuda|macos`, `--extras=a,b,c`, `--non-interactive`, `--yes`, `--no-lock`, `--no-sync`, `--help`. CI: `--non-interactive --no-sync` swaps template + locks, no install.

**Legacy per-platform switchers** (thin wrappers around `setup_pyproject.sh/.ps1`, kept for back-compat):

| Platform | Switch (from repo root) | Then |
|----------|-------------------------|------|
| Linux CUDA 12.8 | `bash bin/use_pyproject_linux_cuda.sh` | `uv sync --extra gpu` and optionally `--extra sam` |
| Windows CUDA 12.1 | `pwsh bin/use_pyproject_win_cuda.ps1` | same |
| macOS (Metal) | `bash bin/use_pyproject_macos_metal.sh` | `uv sync` |

**Back to portable CPU** (e.g. same clone on laptop): `bash bin/use_pyproject_universal_cpu.sh` (Linux/macOS) or `pwsh bin/use_pyproject_universal_cpu.ps1` (Windows), then `uv sync`.

Each switch runs `uv lock`, rewrites `uv.lock` for that hardware matrix. Default lock in git targets **CPU**; CUDA users regenerate locally after switch.

SAM 3 video (`vaila_sam.py`) needs **NVIDIA CUDA** at runtime (`torch.cuda.is_available()`), even with `sam` extra installed. **No** CPU-only or **macOS Metal/MPS** path here; `--frame-by-frame` only lowers **VRAM on CUDA**, not CPU fallback. Without CUDA, use other vailá modules (e.g. Markerless 2D / YOLO) or CUDA workstation/cloud GPU. Checkpoint auto-detect supports both `vaila/models/sam3/` and repo-root `models/sam3/`.

**Sapiens2 Pose (optional):** `uv sync --extra sapiens` plus `bash bin/setup_sapiens2.sh` (clones into `.local/third_party/sapiens2/`, editable install, downloads `facebook/sapiens2-pose-1b` + `facebook/detr-resnet-101-dc5` into `vaila/models/sapiens2/`). GUI: Frame B → **YOLO + FB** → **Sapiens2 Pose** (`vaila/vaila_sapiens.py`). Default model **1B** fits RTX 4090 24 GiB. Help: `vaila/help/vaila_sapiens.md`. License: Meta Sapiens2 License (not AGPL).

**FIFA Skeletal Tracking Light (optional):** `uv sync --extra fifa` (workstation: combine CUDA template + `--extra gpu`). `sam_3d_body/` **not committed** — clone via `bash bin/setup_fifa_sam3d.sh` (or `pwsh bin/setup_fifa_sam3d.ps1` Windows), which also downloads gated `facebook/sam-3d-body-dinov3` weights into `vaila/models/sam-3d-dinov3/`. Vendored MIT starter-kit utils in `vaila/fifa_starter_lib/` (`camera_tracker.py`, `postprocess.py`, `pitch_points.txt`; see `vaila/fifa_starter_lib/VENDOR.md`). CLI: `uv run vaila/vaila_sam.py fifa <subcommand> --help`, subcommands `bootstrap` (symlinks + sequences + pitch_points), `prepare`, `boxes`, `preprocess`, `baseline`, **`dlt-export`** (FIFA `cameras/*.npz` → per-frame `.dlt2d`/`.dlt3d` via `vaila/fifa_to_dlt.py` for **`rec2d.py` / `rec3d.py`** on moving broadcast cameras), `pack`. Use **`rec2d_one_dlt2d.py` / `rec3d_one_dlt3d.py` only for fixed cameras** (single DLT row). Companion tool `vaila/soccerfield_calib.py` (button **Soccer-Field Calib**, Frame C of `vaila.py`) fits **single-frame** DLT2D homography from 29 FIFA keypoints; GUI **FIFA cams→DLT** exports per-frame DLT after `baseline --export-camera`. Tests: `uv run pytest tests/test_fifa_skeletal_pipeline.py tests/test_fifa_bootstrap.py tests/test_fifa_to_dlt.py tests/test_soccerfield_calib.py -v`. Full `data/` layout (`cameras/`, `boxes/`, …) still from official starter kit / Hugging Face dataset when available.

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

Project uses `pytest` for automated testing.

- `tests/test_vaila_and_jump.py` — unit tests, biomechanical calcs.
- `tests/test_vaila_and_jump_integration.py` — integration tests, full analysis pipelines, sample data.
- `tests/vaila_and_jump/` dir has sample data (CSV, TOML) for these tests.

**Milestone (02 March 2026):** Refactored `vaila_and_jump.py` (v0.1.3), `vaila/tugturn.py`, DLT/Reconstruction suite (`dlt2d.py`, `dlt3d.py`, `rec2d_one_dlt2d.py`, `rec3d.py`, `rec3d_one_dlt3d.py`). Fixed all Ruff/Ty lint+type errors, added CLI/headless support, established comprehensive automated test suite across `tests/`.

## Mandatory: Update metadata on any script change

Change **any** Python script (`*.py`) anywhere in repo → also update user-facing metadata so version/date stay consistent.

### Checklist

- **Script header**: in edited `*.py`, update top module docstring/header fields:
  - **Update Date**: today
  - **Version**: **global vailá version** (same as `vaila.py` header/banner)
- **Main entry point**: if change impacts GUI/CLI banner, also update `vaila.py`:
  - header **Update Date** + **Version**
  - any printed/banner strings embedding version/date
- **Installers**: review/update if install/run UX impacted:
  - `install_vaila_linux.sh`, `install_vaila_mac.sh`, `install_vaila_win.ps1`, `install-hooks.sh`
- **README**: update root `README.md` line `Last updated: YYYY-MM-DD` to today
- **Help docs**: keep help in sync with edited script:
  - main index `vaila/help/index.md` + `vaila/help/index.html` ("Generated on")
  - edited module help `vaila/help/<module>.md` + `vaila/help/<module>.html` (Version + Updated)

## External unified pitch dataset (YOLO retrain, outside repo)

Merged **32 pitch keypoint** tree from `vaila.fifa_dataset_builder` designed to live on disk **outside** git clone (large image banks). Ultralytics training points at `<dataset_root>/unified/data.yaml` with **absolute** `data=` path. After QA on flat `check_all_labels/` export, use `vaila.fifa_check_labels_dedupe` and `vaila.fifa_dataset_train_readiness` (`--prune-unified-to-flat`) so `unified/` matches human-validated samples. Narrative: **`docs/fifa_workflow.md` §4.5**; GUI companion help: **`vaila/help/soccerfield_keypoints_ai.md`** (Training → Option B).

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

**`vaila/models/`:** Reference **`.csv`** (similar small files) **tracked**. Downloaded weights (**`.pt`**, **`.ckpt`**, **`.onnx`**, **`.engine`**, **`.task`**, **`.safetensors`**, etc.) and **`vaila/models/**/.cache/`** **gitignored**; fetch via first run or Hub download on **each PC** (see **[docs/huggingface_setup.md](docs/huggingface_setup.md)** — `huggingface-hub>=1.22`, `hf auth login`, lock cleanup for `0.00B` stalls). Examples: [facebook/sam3](https://huggingface.co/facebook/sam3), `facebook/sam-3d-body-dinov3` via `bash bin/setup_fifa_sam3d.sh`. Small default **`.pkl`** (walkway ML) may stay tracked if **< 20 MiB**. Pre-commit blocks staged files **≥ 20 MiB**. Details: [CONTRIBUTING.md](CONTRIBUTING.md#vaila-models-directory). **`tests/SAM/*.mp4`** gitignored (place sample locally; see `tests/SAM/README.md`).

## Platform-Specific Configuration

Project uses **template-based pyproject.toml system** for hardware-specific deps. Before creating venv, correct template must be copied to `pyproject.toml`:

- `pyproject_win_cuda12.toml` — Windows NVIDIA CUDA 12.1
- `pyproject_linux_cuda12.toml` — Linux NVIDIA CUDA 12.8
- `pyproject_macos.toml` — macOS Metal/MPS (Apple Silicon)
- `pyproject_universal_cpu.toml` — CPU-only fallback

Install scripts (`install_vaila_linux.sh`, `install_vaila_mac.sh`, `install_vaila_win.ps1`) handle automatically. Manual setup: copy template **before** running `uv python pin` / `uv venv`.

## Architecture

### Entry Point & GUI (`vaila.py`)

`vaila.py` main entry point. Defines `Vaila(tk.Tk)` class, builds entire GUI. Interface organized into three frames:

- **Frame A (File Manager):** Rename, import, export, copy, move, remove, tree, find, transfer
- **Frame B (Multimodal Analysis):** IMU, MoCap, Markerless 2D/3D, EMG, Force Plate, GNSS, etc.
- **Frame C (Tools):** Data Files (CSV editing, C3D conversion, DLT reconstruction), Video/Image processing, Visualization

Each GUI button dispatches to function in `vaila/` via **lazy imports** — modules imported inside handler methods, avoids loading entire dependency graph at startup. Two dispatch patterns used:

1. **Direct import + call:** `from vaila import module; module.run_function()` — same process
2. **Subprocess launch via `run_vaila_module()`:** Launches `python -m vaila.module_name` separate process — used when Tkinter conflicts could occur (e.g. modules creating own Tk root)

### Package Structure (`vaila/`)

`vaila/` package holds ~100 self-contained analysis modules. Each typically:

- Has `run_*()` or `analyze_*()` entry function called from GUI
- Uses Tkinter `filedialog` to prompt users for input dirs/files
- Reads CSV/C3D via `pandas`/`numpy`/`ezc3d`
- Processes data, writes results (CSV + PNG plots) to timestamped output subdirs

Key shared modules:

- `data_processing.py` — CSV/C3D reading, auto-header detection
- `filtering.py` / `filter_utils.py` — Butterworth, FIR filter implementations
- `common_utils.py` — Header detection, data reshaping for CSV files
- `dialogsuser.py` / `dialogsuser_cluster.py` — Reusable Tkinter input dialogs for sample rate, file type
- `filemanager.py` — All file management operations (rename, copy, move, transfer via SSH)
- `hardware_manager.py` — GPU/CPU detection, TensorRT auto-export for YOLO models
- `interp_smooth_split.py` — Interpolation, smoothing, splitting pipeline (GUI + CLI); configured via `smooth_config.toml`

### Import Conventions

Modules use **relative imports** as part of package (`from .readcsv import ...`), `try/except` fallback to absolute imports for standalone execution (`from readcsv import ...`). Dual-import pattern common throughout codebase.

### Ruff Configuration

Ruff configured in `pyproject.toml`:

- Target: Python 3.12, line length 100
- Enabled rule sets: E, W, F, I, N, NPY, UP, B, C4, SIM
- Ignored: `E501` (line length handled by formatter), `N806`/`N803` (scientific code uses uppercase variable names like X, Y, F)
- `__init__.py` files ignore `F401` (unused imports intentional re-exports)

## Conventions

- **Python version:** 3.12 (strictly `>=3.12,<3.13`)
- **GUI framework:** Tkinter (standard library) — no other GUI frameworks
- **Scientific computing:** Use `numpy`/`pandas` efficiently for data processing
- **Naming:** Scientific code may use uppercase variable names (X, Y, Z, F, etc.) — acceptable, suppressed in linting
- **Output pattern:** Analysis modules write results to timestamped subdirectories (e.g. `processed_linear_lowess_YYYYMMDD_HHMMSS/`)
- **Build system:** `hatchling` backend, managed via `uv`
- **GUI→CLI mirror:** any module with CLI must print copy-paste equivalent commands on GUI **Run** using `>>` prefix (absl logging eats `[bracketed]` stdout). Chooser **YOLO + FB** prints launcher CLI per button; full args in `vaila_sam`, `vaila_sapiens`, `yolov26track track`, `yolotrain`. Reference: `docs/vaila_buttons/yolo-fb.md`, `yolotrain._format_training_cli_command`, `getpixelvideo` `>> Equivalent CLI`.

## History (cross-IDE memory)

Quick lookup for known-hard issues w/ documented fix (full debugging session, hypotheses, runtime evidence, fix details in matching skill under `.claude/skills/`).

### SAM3 batch CUDA OOM cascade (April 2026, debug session 42b4a5)

Symptom: `vaila/vaila_sam.py` batch over directory failed with `CUDA OOM while loading the video into SAM3` on most videos after first OOM, even with 256→64→32 retry ladder, even after `_release_sam3_gpu_memory()` (gc + empty_cache).

Root cause: **two compounding leaks** on OOM path of `predictor.handle_request("start_session")`:

1. Live `Exception e` traceback retained ~7 GiB inner-frame SAM3 tensors. Calling `_release_sam3_gpu_memory()` *inside* `except` block no-op — `e` still alive — release only worked at **top of next iteration** (after `e` implicitly deleted).
2. ~13 GiB orphan tensors held in **SAM3's C++ workspace pools** (CUDA-side state opaque to Python's gc) survived `predictor = None`, `gc.collect()`, `torch.cuda.empty_cache()`. Only releasable by killing Python process.

Fix in `vaila/vaila_sam.py`:

- Moved `_release_sam3_gpu_memory()` to start of each retry-loop iteration in `_process_one_video_with_oom_retry`.
- Added **subprocess-per-video isolation** in CLI batch loop (`main()`): per video, spawn `python vaila_sam.py --input <single.mp4> --video-output-dir <out>` so OS-level process death guarantees clean GPU for next video. Default ON when batch >1 video; disable with `--no-isolate-batch` for debugging only. New internal flag `--video-output-dir` lets child write directly to parent-supplied dir without creating own `processed_sam_TS/` wrapper.
- Added **rich outputs** for downstream multi-camera / 3D reconstruction: overlay draws bbox/ID/score/contours, run exports `sam_contours.json` (polygons), `sam_tracks.csv` (long bbox+area stats), `sam_masks_manifest.csv` (mask index).

Full hypothesis log, runtime evidence, code map: see `.claude/skills/sam3-video/SKILL.md` § *Why subprocess-per-video* and § *Cascading OOM in batch*.

### Sports field CLI and GUI integration fixes (April 2026, session efe5a0)

Symptom 1: `vaila/drawsportsfields.py` crashed with `qt.qpa.plugin` errors run in CLI mode (`--field`) on Linux, while fine launched from main `vaila.py` GUI.
Symptom 2: `App` class init in `vaila/markerless_3d_analysis.py` failed with `ArgumentError` (expected `Tk`, found `Toplevel`) integrated into notebook interface.
Symptom 3: Matplotlib reported `Module matplotlib.cm has no member rainbow` — deprecated attribute access.

Root cause 1: Matplotlib defaulted to `QtAgg` backend in standalone CLI mode. Systems w/ missing/broken Qt deps → immediate crash. Also `--field` flag wrongly triggered full Tkinter GUI loop instead of simple plot.
Root cause 2: `App` class defined inside `if __name__ == "__main__":` block, constructor type-hint restricted to `tk.Tk`, prevented use as `tk.Toplevel` child window.
Root cause 3: Modern Matplotlib requires `plt.get_cmap("name")` instead of `plt.cm.name`.

Fixes:
- Added `matplotlib.use("TkAgg")` before `pyplot` imports in `drawsportsfields.py`, ensures cross-platform compat w/o Qt.
- Refactored `drawsportsfields.py` CLI logic, uses static `plt.show()` instead of full `run_soccerfield()` GUI when args passed.
- Moved `App` class out of `__main__` in `markerless_3d_analysis.py`, updated type hint to `tk.Tk | tk.Toplevel`.
- Added native FIFA layout support via `soccerfield_ref3d_fifa.csv` and `SPORT_REGISTRY["fifa"]`.
- Updated all colormap accesses to use `plt.get_cmap()`.

Full details: see `.claude/skills/sports-field-visualization/SKILL.md`.

### Soccer field AI keypoints (YOLO pose + video CSV) (April 2026, session 254a97)

**Module:** `vaila/soccerfield_keypoints_ai.py` — Ultralytics local weights or Roboflow API; video mode writes `field_keypoints_video.csv`, `field_keypoints_getpixelvideo.csv`, `field_keypoints_overlay_markers.csv`, optional `field_keypoints_overlay.mp4`.

**Training dataset (local YOLO):** `vaila/models/hf_datasets/football-pitch-detection/data/data.yaml` (`kpt_shape: [32, 3]`). Prefer `imgsz=1280`, `mosaic=0`, `erasing=0`, moderate `pose`/`kobj` weights; naive `mosaic=1` + small `imgsz` can collapse all keypoints into tiny cluster while box mAP stays high.

**Handoff for other IDEs / agents:** read `.claude/skills/soccer-field-keypoints-yolo/SKILL.md` for exact train/export/infer commands, weight path resolution (Ultralytics run dir suffix `-N`), CSV semantics.


### Crop Face GUI integration and Help opener (June 2026, session 2026-06-01)

**Module:** `vaila/crop_faces_atletas.py` — athlete face photo cropper by
Abel Gonçalves Chinaglia. Now follows project GUI pattern: select input
photo dir first, then output dir. Wired to **Frame C -> Video
and Image -> C_B_r1_c2 - Crop Face**. Help files at
`vaila/help/crop_faces_atletas.md` and `.html`; `README.md` and
`vaila/help/index.*` list new button.

**Model path:** MediaPipe detector downloaded first use into
Git-ignored `vaila/models/crop_face/face_detector.task`; provision explicitly
w/ `uv run python vaila/crop_faces_atletas.py --download-model`. Legacy
locations `vaila/crop_face/models/face_detector.task`,
`vaila/models/face_detector.task`, repo-root `models/face_detector.task`
still auto-detected. If download fails, GUI asks user select
compatible `.task` or `.tflite` file. Keep downloaded model files out of git.

**Help opener fix:** main `Help` button in `vaila.py` must use
`webbrowser.open_new_tab(Path(...).as_uri())` for `vaila/help/index.html`.
Don't use shell `open`/`xdg-open` through `os.system` for this button; on Linux
that can open IDE/editor instead of browser depending on user file
associations.

### Save-freeze fix + verbose README + sam_bbox_tracks alias (June 2026, v0.3.55)

After prior day's smart-loader work users could **load** SAM3 `sam_tracks.csv`
into `getpixelvideo.py`, convert bboxes → markers, but clicking **Save**
froze pygame GUI 10+ min on long broadcast clips, forcing
`sudo kill -9`. Three issues fixed:

1. **Wrong Save branch.** After smart loader converted bboxes to markers,
   Save handler still hit `elif csv_loaded and tracking_data:`, called
   `export_labeling_dataset`, which extracts every annotated frame to disk
   (`cv2.VideoCapture` seek + decode per frame + 3 file writes per frame).
   For 16 693-frame clip w/ 248 K bboxes = tens of minutes blocking I/O on main pygame thread. **Fix:** new state flag
   `bbox_converted_to_markers` (set in `load_tracking_csv` after successful
   anchor conversion) routes Save through `save_coordinates`
   (regular `*_markers.csv`) instead. Users who *want* YOLO dataset
   export simply answer anchor prompt w/ Enter (keep bboxes overlay
   only); Save then takes dataset path.

2. **`save_coordinates` itself O(N×slots) pandas `.at[]` calls.** Same
   248 K-bbox case = minutes of `df.at[frame, "pN_x"] = …`. Now
   vectorised w/ NumPy bulk assignment into `(n_frames, max_points*2)`
   `float64` array, wrapped in single `pd.DataFrame`. **Benchmark on
   16 693 × 62 grid, 248 310 entries: 0.56 s** (vs. effectively hung).

3. **GUI never told user save in flight.** New
   `_flush_save_message(screen, text)` helper paints yellow banner directly
   to pygame surface, flips display **before** any long save
   begins (both dataset export path and marker save path). No more
   "is it frozen?" panic.

**Discoverability + provenance** (same release):

- New shared helpers in `vaila/vaila_sam.py`: `_write_sam_run_readme()`,
  `_make_sam_bbox_tracks_alias()` (plus `SAM_OUTPUT_FILE_GLOSSARY` constant).
  Both chunked merge path (`_merge_chunk_outputs` caller) and
  single-pass writer (`run_sam3_on_video`) now go through them.
- Every SAM3 run now writes **verbose `README_sam.txt`** — explicit schema,
  units, downstream role for every produced file (`sam_tracks.csv`,
  `sam_frames_meta.csv`, `sam_points.csv`, `sam_id_map.csv`,
  `sam_contours.json`, `sam_masks_manifest.csv`, `<video>_sam_overlay.mp4`,
  `masks/`, `FAILED_sam.txt`). Previous README was 8 lines, only
  named chunk stats.
- Sibling **`sam_bbox_tracks.csv`** created as POSIX hardlink
  (zero disk cost, same inode) or copy fallback. Discoverable name w/
  `bbox` in it; consumers still reading `sam_tracks.csv` unaffected.
  Smart loader detects formats by columns, not filename — both names just
  work.

Version sync: `0.3.55 / 15 June 2026` on `vaila.py`, `vaila/vaila_sam.py`,
`vaila/sam_postprocess.py`, `vaila/getpixelvideo.py`,
`vaila/help/vaila_sam.{md,html}`, `vaila/help/getpixelvideo.{md,html}`.
Tests: 338 passed, 1 skipped, 1 deselected (unrelated `tugturn`
Qt-platform `xcb` env failure).

Skill: `.claude/skills/getpixelvideo-tracking-loader/SKILL.md` § *Save
behaviour (v0.3.55)*. SAM3 helper section:
`.claude/skills/sam3-video/SKILL.md` § *Output Format*.

**Follow-up later same day:** ML dataset writers
(`export_labeling_dataset`, `export_pose_dataset`,
`_export_all_labels_view`) are *legitimately* slow — extract +
re-encode thousands of video frames + write 3 files per frame. Without
terminal output users still assumed GUI hung mid-save. Added three
top-level helpers in `vaila/getpixelvideo.py`: `_save_banner(title, detail)`
(boxed banner w/ destination + counts before each save begins),
`_save_done(message)` (completion tail), `_try_import_tqdm()` (soft import,
`tqdm` already transitive via ultralytics/pytorch). Wired into all three
writers w/ `tqdm` bar per train/val/test split. **Note for future
agents:** absl logging (installed by mediapipe/opencv on import) eats
`[bracketed]` prefixes from stdout — banner uses `>>
vaila/getpixelvideo:` instead. tqdm writes to stderr, unaffected.
Documented in `docs/sessions/2026-06-15-getpixel-savefreeze-readme-bbox-alias.md`
§ 8.

### Cross-Chunk Tracklet Linking + getpixelvideo smart loader (June 2026, v0.3.54)

Two related items shipped together on `reidtrain`. Full transcript:
`docs/sessions/2026-06-14-getpixel-sam3-crosschunk.md`.

**1. `vaila/vaila_sam.py` — Cross-Chunk Tracklet Linking.**

Symptom: when coordinator fell back to `_process_video_chunked` on long
1080p clips (see *Coordinator pattern + chunked fallback* above), each chunk
allocated own local object IDs starting at 1. Merged
`sam_tracks.csv` therefore had random IDs across chunk boundaries, breaking
any downstream Re-ID / trajectory analysis.

Root cause: `_build_cross_chunk_id_maps` called `_assignment_min_cost`, but
that helper never defined in `vaila_sam.py` (assumed importable
from `reid_markers.py`). Result: `NameError` at runtime — entire stitch
silently no-op'd.

Fix: defined `_assignment_min_cost` inline (SciPy Hungarian +
greedy fallback). Wired full 5-step pipeline user specified:
sliding-window overlap of **2 frames** between adjacent chunks,
per-local-id feature cache, bipartite IoU + centroid-distance cost matrix,
Hungarian assignment, linked-list merging w/ `min_iou ≥ 0.05` and
`max_centroid_dist_px ≤ 180` gates. Tunable defaults at
`_build_cross_chunk_id_maps(max_centroid_dist_px=180.0, min_iou=0.05)`; no
CLI flags yet. Test:
`tests/test_vaila_sam.py::test_merge_chunk_outputs` (assertions updated
for new 0-indexed global IDs).

**2. `vaila/getpixelvideo.py` — Intelligent Load Tracking CSV.**

Replaced brittle "must have `Frame` column" loader w/
auto-detection of 5 formats: `sam_tracks`, `sam_frames_meta` (normalised
bbox, converted to pixel using video w/h), `sam_points`, `yolo_multi`
(`all_id_detection.csv`), `yolo_single` (`person_id_NN.csv`). Unknown
files still fall back to legacy YOLO parser.

Added bbox → marker **anchor prompt** via `show_input_dialog`
(`1=center 2=bottom 3=top 4=left 5=right`, Enter = keep overlay only).
Helpers: `_BBOX_ANCHOR_ALIASES`, `_anchor_xy_from_bbox`,
`_detect_frame_col`, `_detect_tracking_format`, `_iter_bboxes_from_df`,
`bboxes_to_marker_coordinates`. Once anchor chosen, bboxes become
regular editable/saveable vailá markers. Skill:
`.claude/skills/getpixelvideo-tracking-loader/SKILL.md`.

Version sync (`0.3.54` / 14 June 2026) applied to `vaila.py`,
`vaila/vaila_sam.py`, `vaila/sam_postprocess.py`, `vaila/getpixelvideo.py`,
`vaila/help/vaila_sam.{md,html}`, `vaila/help/index.{md,html}`, and
`README.md`. Tests pass except unrelated `tests/test_tugturn_integration.py::test_cli_end_to_end` Qt-plugin environment failure (already documented in *Sports field CLI and GUI integration fixes*).

### File Manager Tkinter fixes and hybrid SSH Transfer (July 2026, session 2026-07-01)

**Module:** `vaila/filemanager.py` and `vaila.py`.

Symptom 1: Clicking **Transfer** button on main GUI printed debug messages then immediately closed/crashed entire Python/Tkinter GUI process, exit code 139 (Segmentation Fault).
Symptom 2: Executing `vaila/transfer.sh` via CLI failed w/ `rsync: [sender] change_dir "/mnt/disco2tb1/Downloads" failed: No such file or directory` when user typed/copy-pasted trailing spaces.
Symptom 3: Buttons like **Copy**, **Move**, **Import** caused UI freezes or Tcl/Tk crashes on Linux X11 — created duplicate `tk.Tk()` root instances, started secondary event loops (`root.mainloop()`).

Root cause 1: `_transfer_file_gui()` returned immediately without blocking. Local variables (specifically `StringVar` variables bound to entry fields) went out of scope, Python garbage-collected them. Their `__del__` destructor unregistered variables from Tcl interpreter. When Tk event loop tried to render/process events for widgets bound to those deleted variables, dereferenced NULL pointer, segfaulted.
Root cause 2: Interactive user input in shell script kept trailing whitespace (e.g. `/mnt/disco2tb1/Downloads         `), causing `rsync` to search for non-existent path.
Root cause 3: Python's Tkinter wrapper doesn't support multiple `tk.Tk()` root window loops running simultaneously in same process.

Fixes:
- **Tcl/Tk Segmentation Fault Fix**: Bound `StringVar` instances to window object as attributes, blocked returning from `_transfer_file_gui()` using `root.wait_window(transfer_window)` when run in embedded mode, keeps local scope alive.
- **CLI Trailing Spaces Fix**: Added automatic leading/trailing whitespace trimming to all user inputs in `vaila/transfer.sh` using `xargs`.
- **Duplicate Root Window Fixes**: Refactored `copy_file()`, `move_file()`, `import_file()` to use hybrid window management pattern: detect existing `tk._default_root`, spawn transient modal `tk.Toplevel` dialog, wait using `root.wait_window()`. Only fall back to `tk.Tk()` and `mainloop()` when run standalone outside main GUI.
- **Hybrid GUI-Terminal Transfer**: Because `rsync` needs real interactive terminal (TTY) for SSH password input, Transfer button rewritten to collect params in transient modal GUI dialog, write to temp script, launch script in new terminal emulator window (`gnome-terminal`, etc.) so user can safely type password.

Version sync: `0.3.67 / 01 July 2026` on `vaila.py`, `vaila/filemanager.py`, `vaila/transfer.sh`, `vaila/help/filemanager.{md,html}`, `vaila/help/index.{md,html}`, and `README.md`.
Tests: 430 passed, 1 skipped.
Skill: `.claude/skills/filemanager-tkinter-and-ssh-transfer/SKILL.md`.

### Unified Geometric Re-ID module + full plan implementation (July 2026, v0.3.68)

**New module:** `vaila/geometric_reid.py` — shared Hungarian matching, IoU
helpers, velocity-direction cost, mask IoU, homography gate,
`GeometricFrameLinker` class. Eliminates 3× duplicate `_assignment_min_cost`
and 2× near-identical greedy linker implementations.

**Modules refactored:**
- `vaila/yolov26track.py` — `_GeometricTrackLinker` now alias for
  `GeometricFrameLinker`; greedy replaced by Hungarian; CLI gets
  `--reid-max-gap`, `--reid-max-dist`, `--reid-min-iou`,
  `--reid-direction-weight`, `--reid-homography`, `--appearance-reid`;
  BoT-SORT custom YAML w/ `with_reid + GMC` in CLI via
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

`sam_points.csv` + `sam_id_map.csv` now generated automatically after
every SAM batch or single-video run. No need to pass `--postprocess-points`
explicitly unless skipping via `none`.

**New outputs:** five simple VAILA-style anchor CSVs per video:
`sam_vaila_center.csv`, `sam_vaila_bottom.csv`, `sam_vaila_top.csv`,
`sam_vaila_left.csv`, `sam_vaila_right.csv`. Format: `frame,x1,y1,...,xN,yN`
(one x,y pair per tracked object). Each file uses different bbox anchor
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

**Convention:** modules w/ CLI print copy-paste commands on GUI **Run** using `>>` prefix
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

**Symptom:** CLI and GUI runs left two `processed_sapiens_<timestamp>/` folders — one empty, one w/ CSVs/overlay.

**Root cause:** Default subprocess-per-video isolation spawned workers calling `main()` without `--output-base`. Each child minted **new** timestamp dir before handling `--video-output-dir`, while parent batch already owned real output tree.

**Fix:**
- `main()` returns early when `--video-output-dir` set — no second timestamp folder.
- `_build_isolated_sapiens_cmd()` passes `--output-base` from parent batch to each isolated worker.

**Tests:** `tests/test_vaila_sapiens.py::test_build_isolated_sapiens_cmd_passes_output_base`.

**Docs:** `vaila/help/vaila_sapiens.{md,html}` § Output directory (v0.3.76); session `docs/sessions/2026-07-07-sapiens-output-dir-fix.md`.

Version sync: `0.3.76 / 07 July 2026`.

### SAM3 + Sapiens2 CLI Batch Tuning & GPU Optimization (August 2026, session 2026-08-15)

**Module:** `vaila/sam3sapiens2.py` (SAM3-guided top-down Sapiens2 pose, DETR disabled, retains SAM3 `obj_id`s).

**Headless & CLI Execution:**
- Fully supports headless CLI via `-i /path/to/videos -o /path/to/output --model 1b --pose-batch-size 16`.
- Bypasses Tkinter GUI dialogs completely when `-i`/`-o` (or `--resume`) supplied.
- Operates under text-mode `/usr/local/bin/gpumode --cuda` (`systemctl isolate multi-user.target`), freeing 100% dedicated GPU VRAM from desktop compositing (Xorg/Wayland).

**RTX 4090 (24GB) Benchmarks (Sapiens2 1B, 308 keypoints, 1024x768 crops):**
- Batch 4: 7.93 GiB alloc | 9.57 GiB res | 2.45 crops/s
- Batch 8: 10.33 GiB alloc | 13.54 GiB res | 2.67 crops/s (compute saturated)
- Batch 12: 12.72 GiB alloc | 17.40 GiB res | 2.66 crops/s
- Batch 16: 15.11 GiB alloc | 21.22 GiB res | 2.67 crops/s (recommended max)
- Batch 24: 18.68 GiB alloc | 22.73 GiB res | 2.58 crops/s (<1.3 GiB headroom)

**Key Operational Rules:**
1. **Batch Sizing:** Use `--pose-batch-size 16`. Sapiens 1B compute saturates at batch 8–16; batch 24 gives no extra throughput, risks CUDA OOM during multi-video batches.
2. **Single-Subject Videos:** With 1 person/frame, VRAM footprint stays ~7.19 GiB. Use `--max-persons 1` to filter background false positives.
3. **`--flip-test` (TTA):** Runs 2 forward passes per crop (normal + horizontal flip), doubles Sapiens2 inference time (~2x slower), zero VRAM penalty; recommended for max joint precision.

**Skill:** `.claude/skills/sam3sapiens2-pose/SKILL.md`.

## Caveman mode (optional)

[Caveman](https://github.com/JuliusBrussee/caveman) is skills/plugin pack for AI coding agents (Claude Code, Cursor, Gemini CLI, Windsurf, Copilot, 30+ others). Steers model toward terse replies: fewer filler words/articles, typically **~65–75% fewer output tokens** while keeping technical content intact.

### Install

Upstream installer auto-detects supported agents. **Node.js + `npx`** must be on PATH (installer uses `npx skills add`).

- **macOS / Linux / WSL / Git Bash**

```bash
curl -fsSL https://raw.githubusercontent.com/JuliusBrussee/caveman/main/install.sh | bash
```

- **Windows (PowerShell)**

```powershell
irm https://raw.githubusercontent.com/JuliusBrussee/caveman/main/install.ps1 | iex
```

Re-run same command to refresh. Optional: `--with-init` (see upstream README) to write per-repo rule files into current directory.

### Enable / disable in a session

- **On:** `/caveman`, or phrases like "talk like caveman" / "less tokens please". On Codex: `$caveman`.
- **Off:** `stop caveman` or `normal mode`.

**Intensity:** `/caveman lite` (trim fluff, keep grammar), `/caveman full` (default: minimal articles/sentences), `/caveman ultra` (telegraphic), `/caveman wenyan` (upstream style).

**Auto-clarity:** drop caveman tone for security warnings, irreversible actions, or when user confused; resume after.

**Boundaries:** generated code and formal commit/PR prose can stay normal; style target is conversational assistant output.

### Extra upstream tools (input + workflow)

- **`/caveman:compress <file>`** — rewrite large project memory files (e.g. `CLAUDE.md`) in same terse style, cuts **input** tokens every read (upstream cites ~46% average on those files).
- **`/caveman-commit`** — very short commit messages (focus on *why*).
- **`/caveman-review`** — one-line PR review comments.
- **caveman-shrink (MCP)** — optional proxy compresses MCP tool/resource descriptions, saves system-side tokens.

### Cursor in this repo

Repo-local Cursor rule (always-on terse baseline for IDE): `.cursor/rules/caveman.mdc`. Upstream skills install separate; use `curl` / `irm` commands above.

### In-chat rules (mirror)

Respond terse like smart caveman. All technical substance stay. Only fluff die.

- Drop: articles (a/an/the), filler (just/really/basically), pleasantries, hedging
- Fragments OK. Short synonyms. Technical terms exact. Code unchanged
- Pattern: [thing] [action] [reason]. [next step].
- Not: "Sure! I'd be happy to help you with that."
- Yes: "Bug in auth middleware. Fix:"

## Security

This is **open-source (AGPL-3.0)** repo. Do **not** commit API keys, tokens, credential files. See **[SECURITY.md](SECURITY.md)** and **[CONTRIBUTING.md](CONTRIBUTING.md)**.