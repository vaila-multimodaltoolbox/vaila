# vaila_sam

## Module Information

- **Category:** Multimodal Analysis / Video Segmentation
- **File:** `vaila/vaila_sam.py`
- **Version:** 0.0.4
- **Authors:** Paulo Santiago, Sergio Barroso, Felipe Dias, Lennin Abrão
- **GUI Interface:** Yes (Tkinter batch dialog when no CLI args)
- **CLI Interface:** Yes (`-i`, `-o`, `-t`, ...)
- **Email:** paulosantiago@usp.br
- **GitHub:** [vaila-multimodaltoolbox/vaila](https://github.com/vaila-multimodaltoolbox/vaila)
- **License:** AGPL-3.0

## Description

**vaila_sam** performs video segmentation using **SAM 3** (Segment Anything Model 3) from Meta. It supports text-prompt masks via Hugging Face checkpoints for single or batch video processing.

### Main features:

- Text-prompt video segmentation with SAM 3 / SAM 3.1 Multiplex
- Automatic VRAM management (auto-sizes to GPU memory)
- Batch processing (directory of videos or single file)
- Overlay MP4 output with coloured masks
- Per-frame mask PNG output
- Frame-by-frame fallback for low-VRAM GPUs
- GUI mode (Tkinter dialog) and headless CLI mode
- **FIFA Skeletal Tracking Light** pipeline via `fifa` subcommand

---

## Installation

### 1. SAM 3 Video (standard)

```bash
uv sync --extra sam
```

### 2. Workstation (NVIDIA CUDA template)

After switching to the Linux/Windows CUDA `pyproject` (see `bin/use_pyproject_*`):

```bash
uv sync --extra gpu --extra sam
```

### 3. FIFA Skeletal Tracking Light (optional)

```bash
uv sync --extra fifa
# With CUDA template:
uv sync --extra gpu --extra fifa --extra sam
```

> **Runtime:** SAM 3 video inference needs an **NVIDIA GPU with CUDA**, even if the `sam` extra is installed on a CPU-only machine.

> **Docs:** Hybrid CPU laptop vs NVIDIA workstation — see [`AGENTS.md`](https://github.com/vaila-multimodaltoolbox/vaila/blob/main/AGENTS.md) in the repository root.

### Platform: CUDA, CPU-only, Windows, macOS

| Environment | SAM 3 *video* in vailá |
|-------------|------------------------|
| Linux / Windows + NVIDIA + CUDA PyTorch | Supported |
| Windows **without** NVIDIA CUDA (CPU / integrated only) | **Not supported** — use Markerless 2D, YOLO, or a remote CUDA machine |
| macOS (Apple Silicon MPS / integrated) | **Not supported** for SAM 3 video — same CUDA requirement; use other vailá GPU workflows where applicable (e.g. Metal-backed stacks), or run SAM 3 on a CUDA host |

`--frame-by-frame` (CLI) / GUI “frame-by-frame” reduces **GPU memory** use on CUDA; it does **not** enable CPU inference.

---

## Weights / Checkpoints

### SAM 3 Video Weights (facebook/sam3)

The Hugging Face repo `facebook/sam3` is **gated**. You must accept the license on [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3), then authenticate:

```bash
uv run hf auth login
uv run vaila/vaila_sam.py --download-weights
```

Or download manually:

```bash
uv run hf download facebook/sam3 sam3.pt --local-dir vaila/models/sam3
```

### Checkpoint Resolution Order

The resolver auto-detects checkpoints in this priority:

1. CLI `-w` / `--checkpoint` argument
2. Environment: `SAM3_CHECKPOINT` or `VAILA_SAM3_CHECKPOINT`
3. `vaila/models/sam3/sam3.pt` (flat layout)
4. `vaila/models/sam3/sam3.1_multiplex.pt` (SAM 3.1)
5. `vaila/models/sam3/sam3_weights/sam3.pt` (nested HF local-dir layout)
6. `models/sam3/sam3.pt` (repo-root models layout)
7. `models/sam3/sam3.1_multiplex.pt` (repo-root models layout)
8. `models/sam3/sam3_weights/sam3.pt` (repo-root nested HF layout)
9. Legacy: `<repo>/sam3_weights/sam3.pt` (deprecated)
10. If none found: auto-download from Hugging Face Hub at runtime

> **SAM 3D Body weights are different!** `vaila/models/sam-3d-dinov3/` is for the FIFA Skeletal pipeline, **not** SAM 3 video. Passing a SAM 3D Body checkpoint to SAM 3 video will be rejected.

---

## SAM 3 Video CLI

### Quick Start

```bash
# Single video
uv run vaila/vaila_sam.py -i video.mp4 -o output/ -t person

# Batch (all videos in a directory)
uv run vaila/vaila_sam.py -i videos_dir/ -o output/ -t person

# With VRAM cap
uv run vaila/vaila_sam.py -i video.mp4 -o output/ -t person --max-frames 80

# Download weights only
uv run vaila/vaila_sam.py --download-weights

# Open help in browser
uv run vaila/vaila_sam.py --open-help
```

### CLI Flags

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--input` | `-i` | Path | — | Input video file or directory of videos (batch) |
| `--output` | `-o` | Path | — | Output base directory |
| `--text` | `-t` | str | `person` | Open-vocabulary text prompt for segmentation (no fixed class list) |
| `--frame` | `-f` | int | `0` | Frame index used for the initial prompt |
| `--weights` / `--checkpoint` | `-w` | Path | auto | SAM 3 checkpoint (file or folder); auto-detected if omitted |
| `--max-frames` | — | int | auto | Max frames on GPU (VRAM cap); `0` = full clip |
| `--no-overlay` | — | flag | — | Skip overlay MP4 output |
| `--no-png` | — | flag | — | Skip mask PNG output |
| `--frame-by-frame` | — | flag | — | Process each frame individually (prevents OOM but loses temporal tracking) |
| `--download-weights` | — | flag | — | Download facebook/sam3 into `vaila/models/sam3/` |
| `--open-help` | — | flag | — | Open help page in the browser |

### Invocation Modes

| Mode | How |
|------|-----|
| CLI batch | `uv run vaila/vaila_sam.py -i INPUT -o OUTPUT [-t TEXT]` |
| GUI (Tkinter dialog) | `uv run vaila/vaila_sam.py` (no args) |
| FIFA pipeline | `uv run vaila/vaila_sam.py fifa <subcommand>` |
| Download only | `uv run vaila/vaila_sam.py --download-weights` |

### Output Structure

```
output/
  processed_sam_YYYYMMDD_HHMMSS/
    video_name/
      masks/           (PNG per frame, unless --no-png)
      <video_stem>_sam_overlay.mp4   (unless --no-overlay)
      sam_frames_meta.csv
      README_sam.txt
```

---

## VRAM Management

SAM 3 loads **all session frames** onto the GPU. Long clips can exceed VRAM.

| Strategy | How |
|----------|-----|
| Auto (default) | vailá picks a frame cap from the **currently free** GPU VRAM (see `[SAM3 VRAM]` in terminal) |
| Environment variable | `SAM3_MAX_FRAMES=80` |
| CLI override | `--max-frames 80` |
| Full clip | `--max-frames 0` (large VRAM only) |
| Frame-by-frame fallback | `--frame-by-frame` (loses temporal tracking) |

> **PYTORCH_CUDA_ALLOC_CONF:** vailá auto-sets `expandable_segments:True` to reduce CUDA OOM from memory fragmentation.

### Batch behaviour (CLI and GUI)

- Between videos vailá **drops the predictor** and runs `gc.collect()` + `torch.cuda.empty_cache()` so video #N+1 starts from a clean GPU state.
- If a video still hits **CUDA OOM**, vailá **auto-retries** the same clip with `max_frames=64` and then `max_frames=32` before failing.
- A failed video keeps a small **`FAILED_sam.txt`** in its output folder describing the reason — so empty/missing dirs are no longer silent.
- The GUI shows a **progress window** with per-video status, a scrollable log and a **Cancel** button.

---

## Text Prompt Classes

SAM 3 in this module uses **open-vocabulary text prompts**. There is no fixed closed class list beyond `person`.

### Prompt presets (GUI combobox)

The GUI dialog exposes the following presets in the **Text prompt** combobox. The combobox is editable (`state="normal"`), so you can also type any free-text prompt — these are only suggestions:

- `person`, `player`, `goalkeeper`, `referee`, `coach`
- `ball`, `soccer ball`, `basketball`, `volleyball`
- `crowd`, `car`, `bike`, `dog`, `cat`

From the CLI, pass your prompt with `-t "…"`. Example:

```bash
uv run vaila/vaila_sam.py -i match.mp4 -o out/ -t "goalkeeper" -f 0
```

### Prompt tips

- Use short nouns (1-3 words), lowercase. Commas or `.` are ignored.
- If a generic prompt fails, try a **more specific synonym** (`player` instead of `person`; `soccer ball` instead of `ball`) or use a **scene-specific role** (`goalkeeper`, `referee`, `coach`).
- Sport-specific scenarios tested in vailá:
  - Soccer broadcasts → `player`, `goalkeeper`, `referee`, `soccer ball`
  - Basketball → `player`, `basketball`, `referee`
  - Volleyball → `player`, `volleyball`
- Keep the **prompt frame** on a clean, well-exposed moment (no occlusion, steady lens). Use `-f` (CLI) or the "Prompt frame index" field (GUI) to change it.

---

## GUI overview

When you launch `vaila_sam` without CLI args (or from the vailá main window), two Tk windows may appear:

1. **`SamVideoDialog`** — configuration modal:
    - `Input (dir or file)` + `Output folder` + `sam3.pt / weights (-w)` browsers.
    - **Text prompt combobox** with the presets listed above (editable).
    - `Prompt frame index`, `Save overlay MP4`, `Save mask PNGs`, optional frame-by-frame CUDA fallback.
    - Buttons: `Run`, `Cancel`, **`Help`** (opens this page in your default browser).

2. **`SamBatchProgress`** — per-video status window (batch only):

```text
+-----------------------------------------------------------+
| SAM 3 - batch progress                                   |
+-----------------------------------------------------------+
| 2 / 24 processed                                          |
| [==========--------------------------------------------]  |
| Log:                                                      |
|  [14:21:03] Starting video 3/24: ARG_CRO_000737.mp4       |
|  ...                                                      |
|  [Request cancel]  [Help]            [Close (disabled)]   |
+-----------------------------------------------------------+
```

- `Request cancel` finishes the current video cleanly, then stops.
- `Help` opens this reference page.
- `Close` enables when all videos are done.

---

## FIFA Skeletal Tracking Light

The `fifa` subcommand delegates to `vaila/fifa_skeletal_pipeline.py`, providing a complete pipeline for the [FIFA Skeletal Tracking Light 2026](https://inside.fifa.com/innovation/innovation-programme/skeletal-tracking) challenge: monocular broadcast video to 3D human pose estimation (15 keypoints).

### Challenge Links

- [Starter Kit (GitHub)](https://github.com/FIFA-Skeletal-Light-Tracking-Challenge/FIFA-Skeletal-Tracking-Starter-Kit-2026)
- [Dataset (Hugging Face)](https://huggingface.co/datasets/tijiang13/FIFA-Skeletal-Tracking-Light-2026)
- [Validation Portal (Codabench)](https://codabench.org/competitions/11681/)
- [Test Portal (Codabench)](https://www.codabench.org/competitions/11682/)
- [Kaggle Mirror](https://www.kaggle.com/competitions/fifa-skeletal-light/overview)
- [WorldPose Dataset (paper)](https://eth-ait.github.io/WorldPoseDataset/)

### FIFA Setup

```bash
# 1. Switch to CUDA pyproject template (workstation)
bash bin/use_pyproject_linux_cuda.sh

# 2. Install extras
uv sync --extra gpu --extra fifa --extra sam

# 3. Clone sam_3d_body + download gated SAM 3D Body weights
#    (accept the license on https://huggingface.co/facebook/sam-3d-body-dinov3 first,
#     then `uv run hf auth login`).
bash bin/setup_fifa_sam3d.sh                # Linux / macOS
# pwsh bin/setup_fifa_sam3d.ps1            # Windows

# 4. Download challenge data (cameras, boxes, skel_2d, skel_3d, etc.)
#    Accept access at: https://huggingface.co/datasets/tijiang13/FIFA-Skeletal-Tracking-Light-2026
#    Videos come separately from FIFA (not on HF)

# 5. Bootstrap the data layout (symlink videos + sequences_*.txt + pitch_points.txt)
uv run vaila/vaila_sam.py fifa bootstrap \
  --videos-dir /path/to/FIFA_Challenge_2026_Video_Data/Videos \
  --data-root  /path/to/FIFA/data
```

### Data Layout

```
data/
  cameras/
    MATCH_CLIP.npz          (camera intrinsics + extrinsics per sequence)
  images/
    MATCH_CLIP/
      00001.jpg ...         (frames extracted from videos)
  videos/
    MATCH_CLIP.mp4          (broadcast video per sequence)
  boxes/
    MATCH_CLIP.npy          (bounding boxes per frame per person)
  skel_2d/
    MATCH_CLIP.npy          (2D keypoints, 15 joints)
  skel_3d/
    MATCH_CLIP.npy          (3D keypoints, 15 joints)
  pitch_points.txt          (FIFA pitch calibration points)
  sequences_full.txt
  sequences_val.txt
  sequences_test.txt
```

### FIFA Subcommands

#### 0. `bootstrap` — symlink videos + sequences_*.txt + pitch_points.txt

| Argument | Required | Description |
|----------|----------|-------------|
| `--videos-dir` | Yes | Source folder with raw `*.mp4` |
| `--data-root` | Yes | Target FIFA data root |
| `--val-sequences` | No | Text file with official val split |
| `--test-sequences` | No | Text file with official test split |
| `--no-copy-fallback` | No | Fail instead of copying when symlinks are denied |
| `--overwrite-pitch-points` | No | Replace any existing `pitch_points.txt` |

```bash
uv run vaila/vaila_sam.py fifa bootstrap \
  --videos-dir /data/FIFA/FIFA_Challenge_2026_Video_Data/Videos \
  --data-root  /data/FIFA/data
```

Run this *before* `fifa prepare` when your videos are not yet copied into
`data/videos/` and when the official splits have not been written.

#### 1. `prepare` — Videos to data/videos + data/images

| Argument | Required | Description |
|----------|----------|-------------|
| `--video-source` | Yes | Directory with source video files |
| `--data-root` | Yes | Target data/ directory |
| `--sequences-out` | No | Write sequences list file |
| `--extract-fps` | No | FPS for frame extraction |

```bash
uv run vaila/vaila_sam.py fifa prepare \
  --video-source /path/to/raw_videos \
  --data-root data/
```

> `prepare` does **not** create `cameras/`, `pitch_points.txt`, or official `boxes/` — those come from the starter kit / HF dataset.

#### 2. `boxes` — Generate bounding boxes

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--data-root` | Yes | — | Data directory |
| `--sequences` | Yes | — | Sequences file (.txt) |
| `--source` | No | `yolo` | `yolo` or `sam3` |
| `--yolo-model` | No | `yolo11n.pt` | YOLO model name |
| `--sam3-checkpoint` | No | auto | SAM3 checkpoint for `--source sam3` |
| `--text-prompt` | No | `person` | Text prompt for SAM3 |
| `--max-persons` | No | `25` | Max persons per frame |

```bash
uv run vaila/vaila_sam.py fifa boxes \
  --data-root data/ \
  --sequences data/sequences_val.txt
```

#### 3. `preprocess` — SAM 3D Body to skel_2d / skel_3d

| Argument | Required | Description |
|----------|----------|-------------|
| `--data-root` | Yes | Data directory |
| `--sequences` | Yes | Sequences file (.txt) |
| `--no-skip` | No | Re-process even if skel_2d/skel_3d already exist |

```bash
uv run vaila/vaila_sam.py fifa preprocess \
  --data-root data/ \
  --sequences data/sequences_val.txt
```

#### 4. `baseline` — Camera tracker + 3D pose to submission NPZ

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--data-root` | Yes | — | Data directory |
| `--sequences` | Yes | — | Sequences file (.txt) |
| `--output` | Yes | — | Output NPZ path |
| `--refine-interval` | No | `1` | Pitch-point refinement interval |
| `--export-camera` | No | — | Save per-frame calibrated cameras |
| `--visualize` | No | — | Enable visualization |
| `--calibration-dir` | No | auto | Camera export directory |

```bash
uv run vaila/vaila_sam.py fifa baseline \
  --data-root data/ \
  --sequences data/sequences_val.txt \
  --output outputs/submission_full.npz
```

Run **`baseline` with `--export-camera`** (and `--calibration-dir`) when you need refreshed `cameras/<seq>.npz` on disk for the DLT export below (the default `data/cameras/*.npz` from bootstrap is the initial calibration; the tracker updates `K,R,t,k` during `baseline` in memory — export writes the per-frame result).

#### 5. `dlt-export` — `cameras/*.npz` → per-frame `.dlt2d` / `.dlt3d`

Exports vailá-style DLT files (one row per frame) from FIFA camera NPZ files so you can run **`vaila/rec2d.py`** and **`vaila/rec3d.py`** on broadcast / moving-camera pixels.

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--cameras-dir` | Yes | — | Folder containing `*.npz` with keys `K`, `R`, `t`, `k` |
| `--output-dir` | Yes | — | Where to write `<stem>.dlt2d` and/or `<stem>.dlt3d` |
| `--mode` | No | `both` | `2d`, `3d`, or `both` |
| `--undistort-pixels-dir` | No | — | Optional folder of `*.csv`; writes `pixels_undistorted/` under `--output-dir` |

```bash
uv run vaila/vaila_sam.py fifa dlt-export \
  --cameras-dir data/cameras \
  --output-dir outputs/dlt_per_frame \
  --mode both

# Same via module:
uv run python -m vaila.fifa_to_dlt --input data/cameras --output outputs/dlt_per_frame
```

#### 6. `pack` — Split to submission ZIP

| Argument | Required | Description |
|----------|----------|-------------|
| `--submission-full` | Yes | Full submission NPZ from `baseline` |
| `--data-root` | Yes | Data directory (reads `sequences_{split}.txt`) |
| `--output-dir` | Yes | Directory for the output ZIP |
| `--split` | Yes | `val` or `test` |

```bash
# Validation submission
uv run vaila/vaila_sam.py fifa pack \
  --submission-full outputs/submission_full.npz \
  --data-root data/ \
  --output-dir outputs/ \
  --split val

# Test submission
uv run vaila/vaila_sam.py fifa pack \
  --submission-full outputs/submission_full.npz \
  --data-root data/ \
  --output-dir outputs/ \
  --split test
```

### End-to-End Pipeline

```bash
# Step 1: Prepare videos and extract frames
uv run vaila/vaila_sam.py fifa prepare \
  --video-source /path/to/raw_videos --data-root data/

# Step 2: Generate bounding boxes (YOLO or SAM3)
uv run vaila/vaila_sam.py fifa boxes \
  --data-root data/ --sequences data/sequences_val.txt

# Step 3: SAM 3D Body → 2D/3D skeletons (CUDA required)
uv run vaila/vaila_sam.py fifa preprocess \
  --data-root data/ --sequences data/sequences_val.txt

# Step 4: Run baseline → submission NPZ
uv run vaila/vaila_sam.py fifa baseline \
  --data-root data/ --sequences data/sequences_full.txt \
  --output outputs/submission_full.npz

# Step 5a: Pack validation submission
uv run vaila/vaila_sam.py fifa pack \
  --submission-full outputs/submission_full.npz \
  --data-root data/ --output-dir outputs/ --split val

# Step 5b: Pack test submission
uv run vaila/vaila_sam.py fifa pack \
  --submission-full outputs/submission_full.npz \
  --data-root data/ --output-dir outputs/ --split test

# Step 6: Upload outputs/submission_val.zip to Codabench validation portal
#          Upload outputs/submission_test.zip to Codabench test portal
```

### Submitting to Codabench

1. Run the pipeline to generate `submission_val.zip` and/or `submission_test.zip`
2. Go to [Codabench Validation Portal](https://codabench.org/competitions/11681/) (or [Test Portal](https://www.codabench.org/competitions/11682/))
3. Create an account / log in
4. Navigate to **My Submissions**
5. Upload the corresponding ZIP file
6. Wait for evaluation — the primary metric is **MPJPE** (Mean Per Joint Position Error) in millimetres

> **Submission format:** Each ZIP contains a single `submission.npz` with per-sequence arrays of shape `(n_frames, n_persons, 15, 3)` — 15 keypoints with X, Y, Z coordinates in world space.

---

## Main Functions

### vaila_sam.py

- `run_sam3_on_video()` — Core SAM3 video segmentation on a single video
- `run_sam_video()` — GUI entry: Tkinter batch dialog
- `download_sam3_weights_to_vaila_models()` — Download gated SAM3 from HF
- `_resolve_sam3_checkpoint_file()` — Checkpoint auto-detection
- `_maybe_subsample_video_for_vram()` — VRAM auto-sizing
- `_composite_masks_bgr()` — Overlay coloured masks on frames
- `main()` — CLI entry point

### fifa_skeletal_pipeline.py

- `fifa_prepare()` — Copy/convert videos, extract frames
- `fifa_generate_boxes()` — YOLO or SAM3 bounding box generation
- `fifa_preprocess()` — SAM 3D Body to skel_2d / skel_3d
- `fifa_baseline()` — Camera tracker + LBFGS + smoothen to NPZ
- `fifa_pack()` — Split full NPZ to submission ZIP
- `main_fifa_cli()` — CLI dispatcher for FIFA subcommands (includes `dlt-export`)

---

## Full broadcast pipeline (moving camera): SAM → FIFA cameras → DLT → rec2d / rec3d

Use this when the **camera moves** (typical broadcast). You need **one row of DLT coefficients per video frame**, not a single global homography.

| Script | Camera | DLT file shape | When to use |
|--------|--------|----------------|-------------|
| **`vaila/rec2d.py`** | Moving OK | `.dlt2d` with `frame` + 8 coeffs × **N frames** | Broadcast 2D field plane (Z = 0) |
| **`vaila/rec3d.py`** | Moving OK | one `.dlt3d` per camera, each with **N frames** | Multi-view 3D (pixels must be time-aligned across views) |
| `vaila/rec2d_one_dlt2d.py` | **Fixed** only | one row of 8 coeffs | Tripod / static cam |
| `vaila/rec3d_one_dlt3d.py` | **Fixed** only | one row of 11 coeffs per cam | Static multi-cam lab |

**Steps (outline):**

1. **SAM 3 video (batch):** segment all clips and optional point CSVs, e.g.  
   `uv run vaila/vaila_sam.py -i /path/to/Videos/ -o sam_out/ -t player --postprocess-points`  
   (Batch uses **one subprocess per video** by default so CUDA memory resets between clips; see SAM3 skill / `AGENTS.md`.)

2. **FIFA layout:** `fifa bootstrap` / `prepare` so each sequence has `videos/`, `images/`, starter `cameras/<seq>.npz`, `pitch_points.txt`, etc.

3. **Boxes → preprocess → baseline** as in the FIFA section. For DLT export, run baseline **with** `--export-camera` (optionally `--calibration-dir …`) so per-frame `K,R,t,k` are saved to NPZ files you will convert.

4. **DLT export:**  
   `uv run vaila/vaila_sam.py fifa dlt-export --cameras-dir data/cameras --output-dir dlt_out/ --mode both`  
   Optional: `--undistort-pixels-dir path/to/sam_csvs` if radial distortion is non-negligible (see warning printed when `|k|` is large).

5. **2D reconstruction:**  
   `uv run python vaila/rec2d.py --dlt-file dlt_out/<SEQ>.dlt2d --input-dir sam_csv_folder/ --output-dir rec2d_out/ --rate 30`  
   Pixel CSVs must use the header `frame,p1_x,p1_y,...` and frame indices must match rows in the `.dlt2d` file.

6. **3D reconstruction (multi-camera):**  
   `uv run python vaila/rec3d.py --dlt-files dlt_out/camA.dlt3d dlt_out/camB.dlt3d --input-dir merged_pixels/ --output-dir rec3d_out/ --rate 30`  
   You must supply **time-synchronized** pixel CSVs and one **per-frame** `.dlt3d` per camera. The current `rec3d.py` expects one combined CSV directory — if your data are **one CSV per camera**, use `rec3d_one_dlt3d.py` only when cameras are **fixed**; for moving cameras, merge frames into the layout your analysis expects or preprocess columns accordingly.

**GUI:** main window button **“FIFA cams→DLT”** runs the same Tk flow as `vaila.fifa_to_dlt.run_gui_flow()`.

---

## Warnings and Known Issues

- **CUDA required:** Both SAM 3 video and the FIFA pipeline (`preprocess`, `baseline`) require an NVIDIA GPU with CUDA
- **BFloat16 type mismatch:** Some SAM 3 builds produce `Input type (c10::BFloat16) and bias type (float) should be the same` — vailá patches `autocast` and forces FP32 on the tracker backbone, but edge cases may persist
- **Gated repositories:** Both `facebook/sam3` and `facebook/sam-3d-body-dinov3` require Hugging Face license acceptance; 403 errors mean the account is not authorized
- **VRAM OOM:** Long clips load all frames onto GPU; use `--max-frames` or `SAM3_MAX_FRAMES` to cap
- **BPE vocabulary:** SAM 3 PyPI wheel may omit the CLIP BPE file; vailá falls back to `boxmot`'s copy or the package `merges.txt`

---

## Requirements

- Python 3.12
- NVIDIA GPU with CUDA (inference)
- `uv sync --extra sam` — SAM 3 video segmentation
- `uv sync --extra fifa` — FIFA Skeletal Tracking (Lightning, timm, omegaconf)
- `uv sync --extra gpu` — CUDA template for TensorRT/CUDA wheels
- OpenCV (`cv2`), NumPy, Tkinter
- Hugging Face Hub (`huggingface_hub`) for weight downloads

---

## Testing

```bash
# Fast unit tests (no GPU, no weights)
uv run pytest tests/test_vaila_sam.py -v
uv run pytest tests/test_fifa_skeletal_pipeline.py -v
uv run pytest tests/test_fifa_to_dlt.py -v

# GPU smoke test (needs CUDA + sam3 + test1000.mp4)
export VAILA_TEST_SAM_GPU=1
uv run pytest tests/test_vaila_sam.py -v -k sam3_smoke
```

Place a short MP4 at `tests/SAM/test1000.mp4` for smoke tests (see `tests/SAM/README.md`).

---

## Version History

- **v0.0.4 (April 2026):** `fifa dlt-export` / `vaila.fifa_to_dlt` — FIFA `cameras/*.npz` → per-frame `.dlt2d` / `.dlt3d` for `rec2d.py` / `rec3d.py` (moving broadcast camera); help section **Full broadcast pipeline**; GUI button **FIFA cams→DLT**
- **v0.0.3 (17 April 2026):** `SamVideoDialog` Help button + editable prompt combobox (14 presets); `SamBatchProgress` Help button; `fifa bootstrap` subcommand (symlinks + sequences + pitch_points); `bin/setup_fifa_sam3d.sh/ps1` for SAM 3D Body cloning and gated weights download; `vaila/fifa_starter_lib/` vendorised (MIT); companion **Soccer-field DLT2D calibration** in `vaila/soccerfield_calib.py`
- **v0.0.2 (April 2026):** FIFA Skeletal Tracking Light pipeline integration; SAM 3.1 Multiplex support; BFloat16 patches; VRAM auto-sizing
- **v0.0.1:** Initial SAM 3 video segmentation with text prompts

### Companion tool: Soccer-field 2D calibration

The `vaila/soccerfield_calib.py` script fits a DLT2D homography from manually
clicked soccer-field keypoints (29 FIFA reference points, 105×68 m). Run:

```bash
uv run vaila/soccerfield_calib.py \
  --video ARG_CRO_000737.mp4 \
  --ref3d models/soccerfield_ref3d.csv \
  --data-root /data/FIFA/data        # drops cameras/<stem>_homography.npz
```

The GUI button **"Soccer-Field Calib"** in vailá's main window (Frame C)
launches the same entrypoint. A Z-vertical extension using goalposts and
anthropometric priors is tracked as a `# TODO: Z vertical (DLT3D)` marker in
the script and will ship in a follow-up.

---

Generated: April 17, 2026
Part of vailá - Multimodal Toolbox
[GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
Contact: paulosantiago@usp.br
