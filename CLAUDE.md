# CLAUDE.md

Guidance for **AI Assistants** (Claude Code, Antigravity, Cursor, Windsurf, etc.) when working with the **vailá** repository.

> See also: [AGENTS.md](./AGENTS.md) — shared rules for all AI agents.

## Project Overview

**vailá** (Versatile Anarcho Integrated Liberation Ánalysis) — open-source Python 3.12 multimodal toolbox for biomechanical data analysis. Integrates IMU, motion capture, markerless tracking (MediaPipe, YOLO), force plates, EMG, GNSS/GPS through a Tkinter-based GUI.

- **GitHub:** https://github.com/vaila-multimodaltoolbox/vaila
- **Python:** strictly `>=3.12,<3.13`
- **License:** AGPLv3
- **Build backend:** `hatchling` managed via [`uv`](https://docs.astral.sh/uv/)

---

## Astral Toolchain

The project uses the full [Astral](https://astral.sh) Rust-based toolchain:

| Tool                                   | Purpose                                                    | Replaces                        |
| -------------------------------------- | ---------------------------------------------------------- | ------------------------------- |
| [`uv`](https://docs.astral.sh/uv/)     | Package manager, venv, Python installer                    | pip, poetry, pyenv, virtualenv  |
| [`ruff`](https://docs.astral.sh/ruff/) | Linter + formatter                                         | flake8, black, isort, pyupgrade |
| [`ty`](https://docs.astral.sh/ty/)     | Static type checker (beta, Rust, 10-100x faster than mypy) | mypy, Pyright                   |

> **Never use** bare `pip install`, `black`, `isort`, `flake8`, or `mypy` — always use the Astral equivalents via `uv run`.

---

## Commands Reference

### uv

```bash
# Run the application
uv run vaila.py

# Sync dependencies (reads uv.lock + pyproject.toml)
uv sync                        # default template's dependencies (see note below)
uv sync --extra gpu            # only defined on CUDA templates (tensorrt + nvidia-ml-py)
uv sync --extra sam            # SAM 3 optional stack; video still needs NVIDIA CUDA at runtime
uv sync --extra sapiens        # Sapiens2 Pose (308 kp); bash bin/setup_sapiens2.sh after
uv sync --extra fifa           # FIFA Skeletal Tracking Light (SAM 3D Body + PyTorch Lightning)
uv sync --frozen               # CI mode: fail if lock is outdated

# RECOMMENDED: unified interactive bootstrap (auto-detects OS + NVIDIA + extras)
bash bin/setup_pyproject.sh                                      # Linux / macOS / WSL / Git Bash
pwsh bin/setup_pyproject.ps1                                     # Windows PowerShell
bash bin/setup_pyproject.sh --target=linux-cuda --extras=gpu,sam --yes
# Legacy per-platform wrappers (thin shims):
# bin/use_pyproject_universal_cpu.sh | use_pyproject_linux_cuda.sh | use_pyproject_macos_metal.sh
# bin/use_pyproject_universal_cpu.ps1 | use_pyproject_win_cuda.ps1

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

> `ty` is in **beta** — not a drop-in replacement for mypy/Pyright; different design choices and defaults. Use alongside ruff, not instead of it.

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

Whenever you edit **any** Python script (`*.py`) in this repo, also update metadata so users see consistent **date/version** across app, docs, and help.

### Checklist

- **Edited script header**: update top module docstring/header:
  - **Update Date**: today
  - **Version**: **global vailá version** (same as `vaila.py` header/banner)
- **Main entry point**: if change impacts GUI/CLI banner, update `vaila.py` header and any banner strings.
- **Install scripts**: if install/run UX impacted, review/update:
  - `install_vaila_linux.sh`, `install_vaila_mac.sh`, `install_vaila_win.ps1`, `install-hooks.sh`
- **Repo README**: update root `README.md` line `Last updated: YYYY-MM-DD` to today.
- **Help docs**:
  - main index `vaila/help/index.md` + `vaila/help/index.html` (“Generated on”)
  - changed module help `vaila/help/<module>.md` + `vaila/help/<module>.html` (Version + Updated)

### Writing convention: how to style "vailá"

Write the project name **lowercase and italicized** in prose — `*vailá*` in Markdown, `<i>vailá</i>` in HTML — matching root `README.md`'s canonical `# _vailá_ - Multimodal Toolbox` title and body usage (`_vailá_`). Never bold it (`**vailá**` / `<strong>vailá</strong>`) and never capitalize it ("Vailá"/"VAILA") in prose. An audit on 2026-08-04 found `vaila/help/*.md`/`*.html` overwhelmingly plain/unstyled (106/148 `.md`, ~135/148 `.html`) with only a handful bolded (9/148 each) or already italicized (3–4/148 each) — i.e. no consistent prior norm; italic-lowercase is the standard going forward. Apply it when touching a help page for another reason; a dedicated repo-wide sweep has not been done.

---

## Architecture

### Entry Point & GUI (`vaila.py`)

`vaila.py` defines `Vaila(tk.Tk)`, organized into three frames:

| Frame       | Purpose                                                                             |
| ----------- | ----------------------------------------------------------------------------------- |
| **Frame A** | File Manager — rename, import, export, copy, move, remove, tree, find, SSH transfer |
| **Frame B** | Multimodal Analysis — IMU, MoCap, Markerless 2D/3D, EMG, Force Plate, GNSS          |
| **Frame C** | Tools — CSV editing, C3D conversion, DLT reconstruction, video/image, visualization |

**Lazy imports** are used in all handler methods to avoid loading the full dependency graph at startup.

Two dispatch patterns:

1. **Direct import + call** — runs in the same process
2. **Subprocess via `run_vaila_module()`** — separate process (avoids Tkinter conflicts)

**Button grid** (row/col ids match the code, e.g. `B1_r1_c4`; run `uv run vaila.py` to see it live):

| Area | Buttons |
| --- | --- |
| Frame A (r1) | Rename · Import · Export · Copy · Move · Remove · Tree · Find · Transfer |
| Frame B (r1) | IMU · MoCap Cluster · MoCap Full Body · **Markerless 2D** (coringa: Standard/Advanced/YOLOv26 MediaPipe, Yolo+Markerless_MP, YOLOv26 Tracker/Pose/Seg/Train, SAM 3 video, Sapiens2 Pose, SAM3+Sapiens2 [+Visualize ID], Markerless Hands, MP Angles, Face Mesh, Markerless Live) · **Markerless 3D** (coringa: Standard/Advanced YOLO lift, SAM3+DINOv3 3D [+Visualize ID]) |
| Frame B (r2) | Vector Coding · EMG · Force Plate · GNSS/GPS · MEG/EEG |
| Frame B (r3) | HR/ECG · Vertical Jump · Cube2D · Animal Open Field |
| Frame B (r4) | ML Walkway |
| Frame B (r5) | Ultrasound · Brainstorm · Scout · StartBlock · Pynalty |
| Frame B (r6) | Sprint · tugturn · Soccer Tools (Field KPs AI, Soccer-Field Calib, VEK ElasticKick, FIFA cams→DLT) · Deadlift |
| Frame B (r7) | Treadmill LC (step-based ground-reaction-force workflow, TOML config) |
| Frame C-A (Data Files) | Edit CSV · C3D↔CSV · Smooth & Filter · **DLT/REC 2D-3D** (coringa: Make DLT2D/DLT3D, Rec2D/Rec3D 1DLT + MultiDLT) · ReID Marker |
| Frame C-B (Video/Image) | Video↔PNG · Crop Face · Draw Box · Compress Video · Make Sync file · GetPixelCoord · Metadata info · Merge/Split · Distort · Cut · Resize · YT Downloader · Insert Audio · rm Dup PNG |
| Frame C-C (Visualization) | Show C3D/CSV 3D · Plot 2D/3D · Draw Sports · Stroboscopic |

Full ASCII map with descriptions: `README.md` § *vailá Structure and Interface*; per-button docs: `docs/vaila_buttons/`.

### Package Structure (`vaila/`)

~100 self-contained analysis modules. Each module:

- Has a `run_*()` or `analyze_*()` entry point called from the GUI
- Uses Tkinter `filedialog` for user input prompts
- Reads CSV/C3D via `pandas` / `numpy` / `ezc3d`
- Writes results (CSV + PNG plots) to timestamped output subdirectories

**Key shared modules:**

| Module                                      | Role                                                      |
| ------------------------------------------- | --------------------------------------------------------- |
| `data_processing.py`                        | CSV/C3D reading with auto-header detection                |
| `filtering.py` / `filter_utils.py`          | Butterworth and FIR filter implementations                |
| `common_utils.py`                           | Header detection and data reshaping                       |
| `dialogsuser.py` / `dialogsuser_cluster.py` | Reusable Tkinter input dialogs                            |
| `filemanager.py`                            | File management (rename, copy, move, SSH transfer)        |
| `hardware_manager.py`                       | GPU/CPU detection, TensorRT export — **do not duplicate**. First run per model builds a VRAM-sized `.engine` (2–5 min, cached); Windows/Linux engines coexist in the same folder on dual-boot. |
| `interp_smooth_split.py`                    | Interpolation, smoothing, splitting (GUI + CLI)           |

---

## Platform-Specific Configuration

Copy the correct template to `pyproject.toml` **before** running `uv python pin` / `uv venv`:

| Template                       | Target                          |
| ------------------------------ | ------------------------------- |
| `pyproject_win_cuda12.toml`    | Windows + NVIDIA CUDA 12.1      |
| `pyproject_linux_cuda12.toml`  | Linux + NVIDIA CUDA 12.8        |
| `pyproject_macos.toml`         | macOS Apple Silicon (Metal/MPS) |
| `pyproject_universal_cpu.toml` | CPU-only fallback               |

Install scripts handle this automatically: `install_vaila_linux.sh`, `install_vaila_mac.sh`, `install_vaila_win.ps1`.

> The currently-active template can drift from what other docs claim (it's a plain file copy, not a symlink). Before assuming CPU-vs-CUDA, check directly: `diff pyproject.toml pyproject_universal_cpu.toml` (empty output = CPU is active; otherwise diff against the CUDA/macOS templates to identify which one matches).

---

## Coding Conventions

### Mandatory dual-import pattern

Every module must support both package import and standalone execution:

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
- **Scientific variable names** (X, Y, Z, F, R, T, etc.) are valid — suppressed via ruff `N806`/`N803`
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

Sample data lives in `tests/vaila_and_jump/` (CSV + TOML).

---

## Common Task Recipes

### Add a new analysis module

1. Create `vaila/my_module.py` with `run_my_module()` as entry point
2. Apply dual-import pattern at the top
3. Use helpers from `dialogsuser.py` for user prompts
4. Write results to a timestamped output dir
5. Wire button in `vaila.py` with lazy import
6. Lint and type-check: `uv run ruff check vaila/my_module.py --fix && uv run ty check vaila/my_module.py`
7. Add unit test in `tests/`

### Fix all lint + type issues in one shot

```bash
uv run ruff check vaila/ --fix && uv run ruff format vaila/ && uv run ty check vaila/
```

### Run a module standalone via CLI

```bash
uv run vaila/interp_smooth_split.py -i /path/to/csv_dir -c smooth_config.toml
```

---

## Security

Open-source under **AGPL-3.0** — never commit API keys, tokens, or local credential files. See **[SECURITY.md](SECURITY.md)** and **[CONTRIBUTING.md](CONTRIBUTING.md)**. Use `.env` locally (gitignored); see `.env.example` for a safe template.

---

## Agents and skills

Step-by-step workflows and specialized agent roles are stored in the `.claude/` directory. This structure is intended to be used by any AI assistant (Claude Code, Antigravity, Cursor, etc.).


### Recent GUI Notes

- **Crop Face (`vaila/crop_faces_atletas.py`)**: integrated at **Frame C -> Video and Image -> C_B_r1_c2**. The GUI flow must remain input directory first, output directory second, then automatic official model download into Git-ignored `vaila/models/crop_face/face_detector.task`; manual `.task` / `.tflite` selection remains fallback when network download fails. Provision explicitly with `uv run python vaila/crop_faces_atletas.py --download-model`. Creator metadata remains `Abel Gonçalves Chinaglia`; help docs are `vaila/help/crop_faces_atletas.md` and `.html`.
- **Main Help button**: open `vaila/help/index.html` with `webbrowser.open_new_tab(Path(...).as_uri())`. Avoid `os.system("open ...")` / shell openers for this button because Linux file associations may launch an IDE/editor instead of the browser.
- **Smart Load Tracking CSV (`vaila/getpixelvideo.py`, v0.3.55)**: auto-detects SAM3 (`sam_tracks.csv`, `sam_bbox_tracks.csv` alias, `sam_frames_meta.csv`, `sam_points.csv`) and YOLO (`all_id_detection.csv`, `person_id_NN.csv`) formats, then prompts for a bbox → marker anchor (`1=center 2=bottom 3=top 4=left 5=right`, Enter = keep overlay only). See `.claude/skills/getpixelvideo-tracking-loader/SKILL.md`.
- **getpixelvideo Save no longer freezes after bbox load (v0.3.55)**: state flag `bbox_converted_to_markers` routes Save through `save_coordinates` (vectorised NumPy bulk assignment; 248k-bbox case writes in ~0.5 s) instead of `export_labeling_dataset` which extracted every annotated frame. New `_flush_save_message(screen, text)` paints a "Saving…" banner before any long write. ML-dataset writers (`export_labeling_dataset`, `export_pose_dataset`, `_export_all_labels_view`) now print a `>> vaila/getpixelvideo:` banner + per-split `tqdm` bar to the terminal so the user sees progress even while pygame is blocked. **Gotcha for future code:** absl logging (installed by mediapipe/opencv on import) silently eats `[bracketed]` prefixes from stdout — use `>>` instead. Details: `docs/sessions/2026-06-15-getpixel-savefreeze-readme-bbox-alias.md` § 8.
- **SAM3 verbose README + bbox alias (`vaila/vaila_sam.py`, v0.3.55)**: every run writes a verbose `README_sam.txt` (schema/units/role for every produced file) plus a `sam_bbox_tracks.csv` POSIX hardlink (copy fallback) next to `sam_tracks.csv` so the bbox file is easy to spot. Shared helpers `_write_sam_run_readme()` / `_make_sam_bbox_tracks_alias()` + constant `SAM_OUTPUT_FILE_GLOSSARY`. See `.claude/skills/sam3-video/SKILL.md` § *Output Format*.
- **SAM3 Cross-Chunk Tracklet Linking (`vaila/vaila_sam.py`, v0.3.54)**: chunked path now stitches chunk-local IDs into persistent global IDs via a 2-frame sliding overlap + bipartite IoU + centroid-distance cost matrix solved with Hungarian (SciPy `linear_sum_assignment`, greedy fallback). Helper `_build_cross_chunk_id_maps(min_iou=0.05, max_centroid_dist_px=180.0)`. See `.claude/skills/sam3-video/SKILL.md` § *Cross-Chunk Tracklet Linking* and `docs/sessions/2026-06-14-getpixel-sam3-crosschunk.md`.
- **Unified Geometric Re-ID (`vaila/geometric_reid.py`, v0.3.68)**: new shared module consolidating Hungarian assignment, IoU helpers, velocity-direction penalty, mask IoU, and `GeometricFrameLinker`. Replaces 3× duplicate `_assignment_min_cost` and 2× greedy linkers. YOLO, SAM, and markers all import from it. CLI: `--reid-max-gap`, `--reid-max-dist`, `--reid-min-iou`, `--reid-direction-weight`, `--reid-homography`, `--appearance-reid`. SAM: `--overlap-frames N` (default 2). Markers: `geometric_reid_align_markers_bidirectional()`. `reid_yolotrack.py`: parser fixed for `person_id_01.csv`; headless `run_appearance_reid_on_tracking_dir()`. Tests: `tests/test_geometric_reid.py`, `tests/test_reid_yolotrack.py`.
- **YOLO + FB chooser (`vaila.py`, v0.3.71)**: Frame B button renamed from **YOLO + SAM** to **YOLO + FB**; chooser adds **Sapiens2 Pose** (`vaila/vaila_sapiens.py`, 308 kp, CUDA). Bootstrap: `bash bin/setup_sapiens2.sh`; extra `uv sync --extra sapiens`. Help: `vaila/help/vaila_sapiens.md`.
- **GUI→CLI mirror (v0.3.72)**: modules with a CLI path must print copy-paste commands to the terminal on GUI **Run** (prefix `>>`, not `[bracketed]` — absl eats brackets). Chooser **YOLO + FB** prints launcher CLI on each button; full args print in `vaila_sam`, `vaila_sapiens`, `yolov26track track`, `yolotrain`. See `docs/vaila_buttons/yolo-fb.md`.
- **Sapiens2 output directory fix (v0.3.76)**: `vaila/vaila_sapiens.py` no longer creates an empty second `processed_sapiens_*` folder during subprocess-per-video isolation; isolated workers inherit `--output-base` from the parent batch. See `docs/sessions/2026-07-07-sapiens-output-dir-fix.md`.
- **Default post-processing + VAILA anchor CSVs (`vaila/sam_postprocess.py`, v0.3.69)**: `--postprocess-points` default changed from `none` to `all`; GUI default also `all`. `sam_points.csv` + `sam_id_map.csv` now auto-generated after every SAM run. Five new simple VAILA-style CSVs (`sam_vaila_center.csv`, `sam_vaila_bottom.csv`, `sam_vaila_top.csv`, `sam_vaila_left.csv`, `sam_vaila_right.csv`) with `frame,x1,y1,...,xN,yN` format written alongside. New functions: `write_vaila_anchor_csvs()`, `write_vaila_anchor_csvs_for_batch()`, `VAILA_ANCHORS`. Tests: `tests/test_sam_postprocess.py` (10 new anchor tests).

- **SAM3+DINOv3 3D (`vaila/sam3dinov3.py`, v0.3.92)**: new **markerless 3D** module — Frame B → **YOLO + FB → SAM3+DINOv3 3D**. Reuses the `sam3sapiens2` SAM front-end (imports `load_sam_guidance`, `_guidance_for_frame`, `_contour_mask`, `_pose_bbox_from_sam`, `run_sam_stage`) and swaps the second stage for **SAM 3D Body** (`facebook/sam-3d-body-dinov3`, DINOv3 ViT-H/16+ backbone) via `SAM3DBodyEstimator.process_one_image(bboxes=..., masks=...)` — upstream ViTDet/SAM2 never load, so `person_id == sam_obj_id`. **Gotchas:** (1) SAM 3D Body expects **0/1** masks, but `_contour_mask` returns 0/255 — normalise with `(mask > 0).astype(np.uint8)`; (2) the upstream estimator hardcodes `recursive_to(batch, "cuda")`, so **CPU/MPS cannot work**; (3) `pred_keypoints_3d` is **root-relative** — camera-frame is `+ pred_cam_t` (same as `pred_vertices + pred_cam_t`); (4) without `--focal-px` depth inherits the default FOV `f=sqrt(W²+H²)`. Outputs: MHR70 long/wide CSVs (incl. vailá `p1_x,p1_y,p1_z` rec3d convention), camera CSV, optional `meshes/*.npz`. Help: `vaila/help/sam3dinov3.md`. Tests: `tests/test_sam3dinov3.py`.
- **`bin/setup_fifa_sam3d.sh` / `.ps1` — two real bugs fixed (v0.3.92)**: (1) **clone URL** — the Meta repo is **`facebookresearch/sam-3d-body`** (hyphens); the scripts cloned `sam_3d_body` (underscores), which 404s and makes git prompt for credentials. The **local dir and Python package stay `sam_3d_body`**. (2) **`uv pip install -e sam_3d_body/` cannot work** — upstream ships **no `pyproject.toml`/`setup.py`** (it uses `pyrootutils`). The scripts now install only the runtime deps with `--no-deps` (`mhr yacs omegaconf antlr4-python3-runtime==4.9.3 roma trimesh braceexpand pytorch-lightning torchmetrics lightning-utilities`) so the CUDA torch build is never resolved away; `sam3dinov3.ensure_sam3d_importable()` puts the **checkout root** on `sys.path` at runtime (override with `VAILA_SAM3D_BODY_DIR`). **Gotchas:** `omegaconf` needs `antlr4-python3-runtime==4.9.3` (4.13 raises "Could not deserialize ATN with version 3"); the MHR body model is the PyPI package **`mhr`**; and a bare `sam_3d_body/` clone dir also resolves as an *empty namespace package*, so import it via `importlib` + `getattr`, never `from sam_3d_body import X`.
- **Validated on real data (2026-08-01)**: `sam3dinov3` on `~/data/sep_runcod_01072026/REC3D_COD` (1920x1080, 120 fps COD drills) reusing existing SAM3 runs. Segment lengths thigh 0.387±0.017 m / shank 0.371±0.012 m / shoulder width 0.360 m; **100 %** of core joints reproject inside their own SAM bbox vs **0 %** under a shuffled-ID control. Default FOV gives `focal = sqrt(W²+H²) = 2202.9 px` — pass `--focal-px` for true metric depth.
- **Installers offer the `fifa` extra (v0.3.92)**: `install_vaila_linux.sh`/`_mac.sh`/`_win.ps1` now prompt (GPU-gated on Linux/Windows, install-only disclaimer on macOS) to `uv sync --extra fifa` and optionally run `bin/setup_fifa_sam3d.sh`/`.ps1` right after sync — mirrors the existing Sapiens2 prompt pattern. Windows' `Invoke-VailaUvSync` was refactored to accumulate extras generically (`if ($useX) { $syncArgs += ... }`) instead of an if/elseif combinatorial chain, so adding a 4th extra later is a one-liner.
- **DLT/REC family: header-independent pixel CSVs + real rec3d.py bug fix (v0.3.93)**: `rec3d_one_dlt3d.py`, `rec3d.py`, `rec2d.py`, `rec2d_one_dlt2d.py` no longer inspect pixel-CSV column **labels** — only column **order** matters (col 0 = frame, then x,y pairs), so SAM3/YOLO/MediaPipe-labeled CSVs work without renaming. New shared helpers `rec3d.load_pixel_csv_positional()` / `rec3d.find_common_frames()` (imported by `rec3d_one_dlt3d.py`, which also now imports `rec3d_multicam` from `rec3d` instead of duplicating it). **Real bug found in `rec3d.py`:** `process_files_in_directory` treated every CSV in `--input-dir` as an independent single-camera trial, reusing **the same file's row** as the pixel observation for every camera in a multi-camera reconstruction — mathematically wrong for any `--dlt-files` count ≥ 2 (invisible in the old 1-camera-only test). Rewritten to correlate **N pixel files (one per camera, paired with `--dlt-files` by sorted filename) by common frame**, looking up each camera's own per-frame DLT3D row (the "DLT matrix"), producing one reconstructed output instead of one per input file; `--input-dir` file count must now match `--dlt-files` exactly. `dlt2d.py`: `process_files()` returned `None` on one error path and `[]` on another with no caller guard — could crash; standardized to `[]` + guard. `dlt3d.py`: point range was derived from the pixel file only and assumed present in the REF3D file, raising a raw `KeyError` on mismatch; now matches points by common label (intersection) and requires ≥6 common points. Tests: `tests/test_rec_dlt_header_independence.py` (numeric ground-truth regression test for the multi-camera fix, using deliberately non-standard headers). Help: `vaila/help/{rec3d,rec3d_one_dlt3d,rec2d,rec2d_one_dlt2d,dlt2d,dlt3d}.md/.html`.
- **SAM3+DINOv3 Visualize ID (`vaila/sam3dinov3_visualize.py`, v0.3.95)**: new CPU-only rerenderer mirroring `sam3sapiens2_visualize.py`, Frame B → **YOLO + FB → SAM3+DINOv3 Visualize ID**. Selects one `person_id` (== SAM `obj_id`) from an existing `processed_sam3dinov3_*` run, draws SAM contour fill/outline/bbox + the reprojected MHR70 skeleton (via `sam3dinov3.skeleton_edges`) + an `ID nn z=… m` depth label, and writes an ID-specific output: filtered long CSVs (`keypoints3d`/`keypoints2d`/`camera`, by `person_id` column), the already-per-ID wide CSVs copied as-is (`*_id_NN_mhr70_3d.csv`/`*_mhr70_rec3d.csv`/`*_markers.csv` — no slot lookup needed since `person_id == sam_obj_id`), a filtered `*_predictions.json.gz` (gzip round-trip), filtered `sam_contours.json`, and — when `--save-mesh` was used upstream — a filtered `meshes/*.npz` (one row sliced out of the stacked `obj_ids`/`vertices`/`cam_t` arrays per frame) + copied `mesh_faces.npy`. `source_artifacts/` preserves the full original run. No SAM3/SAM 3D Body weights load. Help: `vaila/help/sam3dinov3_visualize.md`.
- **SAM3+DINOv3 Visualize ID: left/right skeleton colors, anti-aliased fill, Blender mesh export (v0.3.96)**: `sam3dinov3_visualize.py` was monochromatic (whole MHR70 skeleton drawn in the single per-ID hue) — fixed by coloring each joint/edge from its `left-`/`right-` name prefix (`_side_color_bgr`): left=green, right=orange, center/spine=blue (edges only take a side color when both endpoints agree, else center). `SAM_CONTOUR_FILL_ALPHA` was `0.35` despite a comment saying it should match SAM3's `~0.45` composite — corrected to `0.45`, and `cv2.fillPoly` now passes `lineType=cv2.LINE_AA` for a smoother silhouette edge (same two fixes applied to `sam3sapiens2_visualize.py` for consistency). New `export_mesh_sequence()` / `--export-mesh {obj,ply}` writes the selected person's per-frame MHR mesh (vertices + shared `mesh_faces.npy`, with `cam_t` translation applied so the body doesn't reset to the origin every frame) as a Blender-importable sequence — no `bpy`/Alembic dependency; on Blender 4.2+ "Mesh: Stop Motion OBJ" is **not bundled** — it moved to the online Extensions platform under the name **"OBJSequence"** (extension id `stop_motion_obj2`, `File > Import > Sequence Directory` or `Import OBJ Sequence`), installable via `Edit > Preferences > Get Extensions` (requires `Allow Online Access` in `Preferences > System`) or offline via `blender -c extension install-file -r user_default -e FILE.zip`. Only meaningful when the *source* `sam3dinov3.py` run used `--save-mesh`; `sam3sapiens2` (Sapiens2, keypoints only) has no surface mesh to export. GUI adds an "Export mesh sequence (.obj)" checkbox. Tests: `tests/test_sam3dinov3_visualize.py` (11 tests, incl. pixel-level left/right color assertions and OBJ/PLY vertex/face-count/translation checks).
- **DLT/REC family: `--fps`/`--rate` accept fractional Hz (v0.3.94)**: `rec3d_one_dlt3d.py` (`--fps`, GUI `askinteger`→`askfloat`), `rec3d.py` and `rec2d.py` (`--rate`, same GUI change) were `type=int`, rejecting real capture rates like NTSC-derived `119.88012001` (120000/1001) needed for an accurate kinematic timeline — silently forced users to round to `120` instead. Now `type=float` end to end; verified the precise value survives into both the BVH `Frame Time` line and the C3D `POINT:RATE` parameter (`ezc3d`accepts float natively — C3D's rate field always was). Validated against real 3-camera Sapiens2 data (308 markers, `~/data/sep_runcod_01072026/REC3D_COD/rec3d_todo`, SAM3+Sapiens2 named-joint headers): thigh/shank/shoulder-width segment lengths anatomically plausible across all 631 frames. `rec2d_one_dlt2d.py` has no rate argument (doesn't export C3D/BVH), so nothing to fix there. Tests: `tests/test_rec_dlt_header_independence.py::test_rec3d_one_dlt3d_accepts_fractional_fps` (+ `rec3d`/`rec2d` variants).
- **GUI reorg: Markerless 2D/3D as coringa choosers, DLT/REC toolkit merge (`vaila.py`, v0.3.97)**: the standalone **Yolo + Markerless_MP** (`B3_r3_c2`) and **YOLO + FB** (`B4_r4_c1`) buttons are gone — their sub-tools now live inside the **Markerless 2D** (`B1_r1_c4`, `markerless_2d_analysis`) and **Markerless 3D** (`B1_r1_c5`, `markerless_3d_analysis`) choosers, split by whether the tool is 2D-only or 3D-native: Markerless 2D gained Yolo+Markerless_MP, YOLOv26 Tracker/Pose(video)/Pose(tracking)/Seg/Train, SAM 3 video, Sapiens2 Pose, SAM3+Sapiens2, SAM3+Sapiens2 Visualize ID; Markerless 3D gained SAM3+DINOv3 3D and SAM3+DINOv3 Visualize ID (the only two SAM3/Sapiens2/DINOv3-family tools that are natively 3D). Separately, the six DLT/REC buttons (`C_A_r2_c1..c3`, `C_A_r3_c1..c3`: Make DLT2D, Rec2D 1DLT/MultiDLT, Make DLT3D, Rec3D 1DLT/MultiDLT) merged into one **DLT/REC 2D-3D** button (`C_A_r2_c1`, `dlt_rec_toolkit`) with the six unchanged handlers as sub-buttons — frees a full grid row in Frame C-A. No underlying script/CLI changed; `_print_chooser_launch()` (renamed from `_print_yolo_fb_launch`) prints the same copy-paste CLI mirror from all three choosers. **Grid alignment gotcha:** Frame B rows are separate `tk.Frame`s packed with `.pack(side="left", expand=True, fill="x")` per button — removing a button from only *some* rows makes that row's buttons stretch wider than aligned rows above/below (fewer buttons sharing the same row width). Fix: every vacated slot (`B3_r3_c2`, `B4_r4_c1`, and the 5 freed `C_A_r2_c2/c3` + `C_A_r3_c1/c2/c3` grid cells) is filled with a standard blank **`vailá`** placeholder button (`command=self.show_vaila_message`), matching the convention already used elsewhere (`B6_r7_c1/c2/c4/c5`, `C_A_r4_c2/c3`, `C_C_r4-r5`) — never leave a chooser-merge slot empty in a `pack(fill="x")` row. **Real bug fixed in passing:** the `C_A_r3_c3` "Rec3D MultiDLT" handler (`rec3d(self)`) was a dead `pass` stub — silently did nothing when clicked — now calls `rec3d.run_rec3d()` like its siblings always did. `vaila/vaila_cli_menu.py`'s `VAILA_MENU_ENTRIES` and `vaila/vaila_cli_hints.py`'s per-handler hints were updated to match (removed entries/hints for the retired codes, added a combined hint per new chooser); `tests/test_vaila_cli_menu.py` numbering (`_NUMBER_BY_CODE`) shifted since two Frame B codes and five Frame C_A codes were removed — recompute via `_ordered_menu_entries()` rather than hand-counting if this table changes again.
- **Markerless 2D/3D dialogs: 3-column scrollable grid + 4 more 2D tools absorbed (`vaila.py`, v0.3.98)**: the choosers were a single tall column of buttons (Markerless 2D alone reached ~900px) — unusable on small-resolution monitors. New shared helper `Vaila._build_grid_chooser_dialog(title, width=, height=, columns=3)` builds a `Toplevel` with a `tk.Canvas` + inner `tk.Frame` scrolled by both a vertical and a horizontal `ttk.Scrollbar`, plus mouse-wheel bindings (`<MouseWheel>`/`<Shift-MouseWheel>` for Windows/macOS, `<Button-4>/<Button-5>` for Linux, unbound on `<Destroy>` so they don't leak onto the main window) — returns `(dialog, place_button, place_section)`; `place_button(text, command, width=)` auto-wraps left-to-right across `columns` then down, `place_section(text)` inserts a full-width header and starts a new row. Both `markerless_2d_analysis` and `markerless_3d_analysis` now use this instead of `.pack()`ing buttons one per line. Separately, **Markerless Hands** (`B4_r4_c3`, `vaila/mphands.py`), **MP Angles** (`B4_r4_c4`, `vaila/mpangles.py`), **Markerless Live** (`B4_r4_c5`, `vaila/markerless_live.py`), and **Face Mesh** (`B5_r6_c2`, `vaila/mp_facemesh.py`) — all 2D-only MediaPipe pipelines — moved into a new "Other 2D tools" section of the **Markerless 2D** chooser; their old standalone buttons became `vailá` placeholders (same alignment-preserving pattern as the earlier YOLO+FB/DLT-REC merges). Their handler methods (`markerless_hands`, `mp_angles_calculation`, `markerless_live`, `face_mesh_analysis`) are unchanged, just called from a new place. **Pre-existing CLI-hint bugs fixed in passing:** `vaila/vaila_cli_hints.py` had wrong script filenames for three of these — `markerless_hands.py`, `mp_angles_calculation.py`, and `face_mesh_analysis.py` don't exist (real files are `mphands.py`, `mpangles.py`, `mp_facemesh.py`); the dead per-handler hint entries were removed and the correct commands folded into the combined `markerless_2d_analysis` hint.
- **Correctness pass on the 4 tools absorbed above (v0.3.98)**: `markerless_live.py` had a real crash bug (bbox-drawing `for box in boxes:` loop was unindented one level too far, so a keypoints-only YOLO result with no `.boxes` raised `UnboundLocalError` and killed the whole live camera loop — fixed indentation) plus a silent failure (no `cap.isOpened()` check after `cv2.VideoCapture`) and per-frame debug `print()`s heavy enough to visibly lag a real-time loop (now gated behind a `DEBUG` flag). `mphands.py` created a second `tk.Tk()` root inside `select_video_file()` — unsupported now that it's called from within the already-running main vailá app — removed; also fixed a `ZeroDivisionError` risk (`fps == 0`) and wrapped the capture/writer/CSV loop in `try/finally` to stop a leak on mid-stream exceptions. `mpangles.py`/`mp_facemesh.py`: 8× bare `except:` → `except Exception` (bare except was swallowing `KeyboardInterrupt`), plus the same VideoCapture/VideoWriter leak pattern fixed with `try/finally`. All four bumped to the current global version.
- **`sam3dinov3.py` live overlay was monochromatic — same left/right fix as the visualize scripts (v0.3.98)**: `_draw_pose_overlay()` (the overlay video `sam3dinov3.py` itself writes during a GPU run, *not* the separate `sam3dinov3_visualize.py` rerenderer that already got this fix in v0.3.96) drew the **entire skeleton in one solid `_color_for_id()` color per person** — no left/right distinction, unlike MediaPipe/Sapiens2-style overlays. Moved the `COLOR_LEFT_RGB`/`COLOR_RIGHT_RGB`/`COLOR_CENTER_RGB` palette + `_rgb_to_bgr()`/`_side_color_bgr()` helpers (name-prefix based: `left-`/`right-` else center) from `sam3dinov3_visualize.py` **into `sam3dinov3.py`** (the module that actually owns `MHR70_NAMES`), and `sam3dinov3_visualize.py` now imports them back — single source of truth, so the live-run overlay and the rerenderer are guaranteed to look identical. `_draw_pose_overlay()` gained a `names: list[str]` parameter (already available at the call site as `keypoint_names(len(MHR70_NAMES))`); edges take a side color only when both endpoints agree (else center), matching the rerenderer's rule. `_color_for_id()` is still used, but now only for the `ID nn z=… m` text label, not the skeleton. Tests: `tests/test_sam3dinov3.py::test_side_color_helper_maps_prefixes_to_palette` / `test_draw_pose_overlay_colors_left_and_right_differently` (new); `tests/test_sam3dinov3_visualize.py`'s two color tests updated to read the constants from `sam3dinov3` instead of a since-removed local copy.

### Specialized Agents (`.claude/agents/`)

Role cards for domain experts. Use these when the task fits their specific domain:

- [biomechanics-analyst.md](.claude/agents/biomechanics-analyst.md)
- [gui-developer.md](.claude/agents/gui-developer.md)
- [video-processor.md](.claude/agents/video-processor.md)
- [test-writer.md](.claude/agents/test-writer.md)

### Technical Skills (`.claude/skills/`)

Reusable "how-to" guides for complex workflows:

- **vailá Core**: [create a new analysis module](.claude/skills/create-analysis-module.md), [port a MATLAB algorithm](.claude/skills/port-matlab-algorithm.md), [getpixelvideo-tracking-loader](.claude/skills/getpixelvideo-tracking-loader/SKILL.md) — smart Load Tracking CSV (SAM3 / YOLO auto-detect + bbox → marker anchor prompt), [yolo-fb-gui-cli](.claude/skills/yolo-fb-gui-cli/SKILL.md) — **YOLO + FB** chooser + GUI→CLI terminal mirror (Cursor CLI resume).
- **Sports AI**:
  - [sam3-video](.claude/skills/sam3-video/SKILL.md) — SAM 3 text-prompt video segmentation, GUI help button, prompt presets, **Cross-Chunk Tracklet Linking (v0.3.54)**.
  - [fifa-skeletal-tracking](.claude/skills/fifa-skeletal-tracking/SKILL.md) — FIFA 2026 pipeline (`fifa bootstrap` / `prepare` / `boxes` / `preprocess` / `baseline` / **`dlt-export`** / `pack`), `vaila/fifa_to_dlt.py` (per-frame DLT for **`rec2d.py`/`rec3d.py`** vs fixed-cam **`rec2d_one_dlt2d.py`**), vendored `fifa_starter_lib`, gated SAM 3D Body setup, soccer-field DLT2D calibration.
  - [soccer-field-keypoints-yolo](.claude/skills/soccer-field-keypoints-yolo/SKILL.md) — Ultralytics YOLO **pitch** pose (32 kp), external merged `unified/` tree, `yolo pose train`; see **`docs/fifa_workflow.md` §4.5** and `vaila/help/soccerfield_keypoints_ai.md`.
- **Reports**: [xlsx](.claude/skills/xlsx/SKILL.md) (Excel), [pdf](.claude/skills/pdf/SKILL.md), [pptx](.claude/skills/pptx/SKILL.md) (PowerPoint).
- **Automation**: [mcp-builder](.claude/skills/mcp-builder/SKILL.md) (Model Context Protocol), [webapp-testing](.claude/skills/webapp-testing/SKILL.md).
- **Visualization**: [web-artifacts-builder](.claude/skills/web-artifacts-builder/SKILL.md).

### FIFA Skeletal Tracking Light 2026

vailá ships a complete pipeline for the
[FIFA Skeletal Tracking Light 2026](https://inside.fifa.com/innovation/innovation-programme/skeletal-tracking)
challenge. The one-line setup is:

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

Companion tool `vaila/soccerfield_calib.py` (button **Soccer-Field Calib** in
Frame C of `vaila.py`) fits a DLT2D homography from 29 FIFA keypoints and can
emit `cameras/<stem>_homography.npz` as a fallback when a sequence has no
official `cameras/*.npz`.

**External unified pitch dataset (YOLO retrain):** `vaila.fifa_dataset_builder` writes `unified/data.yaml` under a user-chosen root **outside** git. After QA on `check_all_labels/`, use `vaila.fifa_check_labels_dedupe` and `vaila.fifa_dataset_train_readiness` to align `unified/`, then `yolo pose train data=/ABS/.../unified/data.yaml`. Full recipe: **`docs/fifa_workflow.md` §4.5**.

### Slash Commands (`.claude/commands/`)

Specs for common shortcuts like `/check` or `/new-module`.
- **rec3d Blender export was MIRRORED — real left/right bug (v0.3.99)**: `rec3d.save_rec3d_as_bvh(swap_yz=True)` wrote `(x, z, y)` and `mesh_alignment.apply_blender_yz_swap()` did the matching column swap. **A bare Y/Z column swap is a reflection (det −1)**, and Blender's BVH *and* OBJ importers then apply their own Y-up→Z-up **rotation** (det +1) on top — net det −1, so the athlete appeared **mirrored in Blender: anatomical left and right swapped**, silently inverting every asymmetry conclusion (push-off limb, limb dominance, COD plant leg). Correct conversion is `(x, y, z) → (x, z, −y)`, a proper rotation (−90° about X, det +1) that round-trips through the importer back to the original `(x, y, z)`. The face-winding reversal in `apply_blender_yz_swap()` was **removed** — it existed only to cancel the reflection's normal flip, and a rotation preserves outward normals by itself. Measured: the marker labelled `left_shoulder` is anatomically left in **100%** of 379 fast frames in the CSV but **0%** after the old export; **95.1%** after the fix (measured in-scene; misses are low-speed COD frames where velocity is a poor proxy for facing). BVH joints now reproduce the reconstruction CSV in Blender world space to **0.00 mm** (was 9005.67 mm). **Gotcha:** the OBJ sequence must be imported with Blender's **DEFAULT** axes (`forward=-Z, up=Y`) — all 30 combinations were probed and the default gives 0.15 m mean mesh-to-skeleton distance vs 2.1–4.6 m for every alternative. Setting `up=Z` "because the data is Z-up" reintroduces a multi-metre offset. Tests: `tests/test_rec3d_occlusion_handling.py` (BVH round-trip + handedness), `tests/test_rec3d_mesh_alignment.py` (det +1, importer round-trip, winding preserved).
- **Occlusion handling across all three rec3d export formats (v0.3.99)**: markerless input (SAM3+Sapiens2 308 kp / SAM3+DINOv3 MHR70) carries NaN for occluded or low-confidence keypoints, and each format needed a different fix. **C3D** (`readcsv_export.py`): `np.nan_to_num(nan=0.0)` wrote occluded samples as 0.0 with a *valid* residual, so viewers drew 30 588 marker-samples (15.7 %) as real markers piled on the world origin — now `meta_points.residuals = -1` (the C3D invalid-sample convention; `ezc3d` round-trips it and returns NaN coordinates on read). **BVH** (`rec3d.py`): has no invalid-sample convention, so 0.0 teleported joints to the origin mid-motion — now interior gaps are linearly interpolated and head/tail gaps hold the nearest valid sample. **Mesh** (`rec3d_one_dlt3d.py`): `reconstruct_mesh_sequence()` dropped a frame if **any** of the 9 alignment markers was NaN, leaving 36 holes in a 631-frame OBJ sequence — which silently **desynchronises** a Blender OBJ-sequence import (it maps files to consecutive frames, so the mesh ends up 36 frames ahead of the BVH). But `best_camera_alignment()` **already** excludes non-finite rows per camera, and on real data only the two acromions ever go missing, leaving 7 good correspondences — the guard was pure loss. Removed it (631/631 frames now solved, 0 interpolated); added `interpolate_similarity_transform()`/`slerp_rotation()` as a per-camera fallback for genuinely unsolvable frames (transforms map *that camera's* monocular space to world, so they are **not** interchangeable across cameras) plus a `sequence_contiguous` report. Largest mesh-centroid step fell from 0.695 m (83 m/s) to 0.103 m (12.3 m/s). Also new `rec3d.find_unreconstructed_markers()`: markers NaN for the whole trial (23 finger keypoints here) have nothing to fill from and stay at the origin, so `generate_blender_companion_script()` now **drops the 15 skeleton connections touching them** instead of drawing bones from the hands to (0,0,0).
- **MHR70 == the first 70 of Sapiens2 Goliath-308 (verified 2026-08-04)**: name-by-name check confirms all 9 `mesh_alignment.ALIGNMENT_MARKER_SPEC` entries (p6/p7 shoulders, p10/p11 hips, p12/p13 knees, **p68/p69 acromions, p70 neck**) match exactly between the two layouts; the 42 "differences" in the first 70 are naming variants only (`left_big_toe_tip` vs `left_big_toe`, `right_thumb_tip` vs `right_thumb4`). **Practical consequence:** a DLT reconstruction built from Sapiens2 308-marker pixel CSVs can drive SAM3+DINOv3 mesh alignment with the default `ALIGNMENT_MARKER_INDICES` — no remapping needed. Validated end to end on `rec3d_todo/sam3sapiens2_one_person_visualized` (3 cameras, 631 frames, 119.88012001 Hz) with `--mesh-source-dir` pointing at the sibling `sam3dinov3_one_person_visualized`: residual mean 0.0265 m / max 0.0437 m, thigh 0.42 ± 0.02 m, shank 0.41 ± 0.02 m, shoulder width 0.35 m.
- **Blender companion script no longer needs the "Stop Motion OBJ" add-on (v0.3.99)**: `rec3d.generate_blender_companion_script()`'s `import_mesh_sequence()` used to call `bpy.ops.import_scene.obj_sequence` and, when the extension was absent (it is **not** bundled with Blender), just printed a hint and returned — so the mesh was silently never imported and users imported the `.obj` files by hand, usually picking **`up = Z`**, which leaves the body only ~0.65 m "tall" and sliding ~2.2 m along Z across the take: it looks like *the mesh sinking through the floor*. The mesh files are written **Y-up** (matching what Blender's OBJ importer expects), so the file→world conversion is `(X, Y, Z) → (X, -Z, Y)` — the same one the BVH importer applies to the skeleton. Rewritten to be add-on-free: since every frame shares one topology, the script now parses the OBJs itself (applying that conversion in Python, so no importer axis setting can get it wrong), builds **one** mesh named `Vaila_Mesh`, and swaps vertex positions from a `frame_change_post` handler (`_apply_mesh_frame`/`_vaila_mesh_frame_handler`, unregistered first so re-runs don't stack). Only one frame of geometry is ever live instead of 631 objects. **Also reordered `main()`:** `setup_scene()` now runs **LAST** (after BVH import, mesh import and bone building) because both the BVH importer and any C3D importer rewrite the scene rate/frame range — whatever runs later gets the final word, and that was the cause of "BVH plays slow and stops early" (scene left at 24 fps / frame_end 250) even when a C3D in the same scene played complete. New `report()` prints the achieved rate/range/armature/mesh with explicit WARNINGs. Re-running the script is now the documented fix for an already-broken scene. Verified headless: rate 119.880114 fps, range 1..631, mesh tracks the skeleton within 0.206 m and stays 1.15–1.79 m tall across the whole take.
- **`rec3d.py`/`rec2d.py` CLI hung on unguarded Tk dialogs — fixed, no longer "pre-existing, out of scope" (v0.3.99)**: both modules called `messagebox.showinfo()` on the success path and `messagebox.showerror()` on every validation-failure path with **no `gui=` guard**. On a machine with a real DISPLAY this **blocks forever** waiting for a click nobody will give (without a DISPLAY it raises `TclError`), so any headless/CLI/subprocess run of `vaila/rec3d.py` or `vaila/rec2d.py` hangs instead of exiting. This was annotated as a TODO earlier in the same session and deferred — it then **hung the test suite** (`tests/test_dlt_rec_integration.py::test_rec3d_integration` and `tests/test_rec_dlt_header_independence.py::test_rec3d_rejects_camera_count_mismatch` both `subprocess.run()` these CLIs with no timeout), which is how it surfaced. Fixed properly: `gui` is now threaded through `run_rec3d`/`run_rec2d` → `process_files_in_directory` → `save_rec3d_as_bvh`, both `__main__` blocks set `gui=not cli_mode` (where `cli_mode` = all required args present), and new `rec3d._report_error(message, gui=True)` always prints and only opens a dialog in GUI mode. **Gotcha for future work in these files:** the GUI-only branch of `run_rec3d` (dialogs for DLT files/dirs/rate/`swap_yz`/skeleton) is fine as-is because it only runs when `dlt_files is None`; the dangerous ones are the calls *after* the GUI/headless split. Suite went from hanging to `76 passed in 65s`.
- **Companion script self-locates its BVH/mesh after a moved or copied run folder (found and fixed by a separate Cursor session, verified 2026-08-05)**: `generate_blender_companion_script()` records the run folder's *absolute* path at export time; a real 631-frame run's script was found pointing at a stale `/tmp/...` scratch directory that happened to still exist, so it silently imported an outdated copy instead of the actual output folder. Fixed with `_resolve(recorded, name, is_dir=False)` in the generated script: prefer the recorded absolute path, else fall back to a same-named file/dir next to the script itself (`_script_dir()`, with a `bpy.data.texts` fallback for Blender's Text Editor where `__file__` is undefined). Also added `README_mesh_import.txt` (written next to every `meshes_<fmt>/`) restating the required manual-import axes. Regression test: `tests/test_rec3d_blender_alignment.py::test_companion_script_finds_its_files_after_the_run_folder_moves` (moves the folder, deletes the original, asserts `_resolve()` still finds both files).
- **Real mesh axis-convention bug found and fixed, reported as "o obj está com o z no lugar do y" on a real run (2026-08-05)**: the 2026-08-04 fix above (mesh swapped in step with `--swap-yz`, matching the BVH file's own raw convention) assumed every OBJ consumer applies the same Y-up→Z-up conversion Blender's BVH importer applies by default. That holds for Blender's own `wm.obj_import` dialog (verified: 0.15–0.21 m mesh-to-skeleton offset with default axes) but **not** for the bundled "Stop Motion OBJ"/OBJSequence family of mesh-*sequence* add-ons — reading its source (`stop_motion_obj2/core.py`'s `parse_obj()`/`apply_to_mesh()`) shows it assigns `"v x y z"` straight to `mesh.vertices.foreach_set("co", ...)` with **no axis conversion at all**, and its import operator exposes no forward/up parameter to override this. So a swapped mesh file landed correctly one static frame at a time via `wm.obj_import`, but through the add-on almost everyone actually plays a *sequence* with, its 1.34 m "tall" dimension landed on Blender's Y axis instead of Z (numerically reproduced: raw-passthrough of a swapped `frame_000300.obj` put the mesh centroid at Z ≈ −2.2 m, nowhere near the skeleton). **Fixed by writing the mesh always in the raw `(x, y, z)` DLT/world frame, unconditionally, regardless of `--swap-yz`** — the same convention the skeleton ends up in once Blender's BVH importer converts it — so the no-configuration add-on now works out of the box, at the cost of `wm.obj_import`'s manual dialog needing an explicit override to `Forward Y, Up Z` (previously "leave the defaults"; `README_mesh_import.txt` and the companion script's own `_read_obj()`, now a plain passthrough, both updated to match). `apply_blender_yz_swap()` itself is unchanged and still correct — it just isn't called for the mesh anymore. Regression test flipped: `tests/test_rec3d_mesh_export.py::test_mesh_vertices_stay_raw_regardless_of_swap_yz` now asserts column 2 (not column 1) holds the plausible-height band. **Separately found and removed:** the same reported run's companion-script *generator* (`rec3d.py`, not the standalone diagnostic script) had a debug-instrumentation block left in from an interactive Cursor session — a `_agent_log()` helper hardcoding `/home/preto/data/vaila/.cursor/debug-e9fe09.log` and wrapping `main()` in a try/except that logged to it — meaning **every** future generated Blender companion script, for any user, would silently attempt to write telemetry to a path that only exists on one machine. Removed entirely (`_agent_log` definition, both call sites, the try/except wrapper around `main()`); the script is back to a plain `main()` call. **Also found (Blender-importer limitation, not a vailá bug):** Blender's bundled C3D importer (`io_anim_c3d`) does not reliably update the scene frame rate from the file even with its own `adapt_frame_rate=True` default — tested empirically in headless Blender against this exporter's own C3D (which correctly states `POINT:RATE=120.0` and header `frame_rate=120.0`): scene stayed at 24 fps after `bpy.ops.import_anim.c3d(...)`. The companion script's `setup_scene()` already runs last regardless of import order, so re-running it after a manual C3D import already corrects the scene rate — documented in the module docstring, README, and help docs rather than worked around, since there is nothing to fix in the exporter itself. **Also found (environment issue on the reporting machine, not a vailá bug):** the same Blender install has the Stop Motion OBJ2 extension installed **twice** (`bl_ext.blender_org.stop_motion_obj2` and `bl_ext.user_default.stop_motion_obj2`), which collide on class registration (`ValueError: register_class(...): already registered as a subclass 'SMO_SequenceProperties'`) and can leave the add-on's operators unregistered after an enable/disable cycle — worth uninstalling one copy if `import_scene.obj_sequence` intermittently "could not be found."
