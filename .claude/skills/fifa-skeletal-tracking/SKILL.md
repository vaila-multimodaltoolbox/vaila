# FIFA Skeletal Tracking Light (vailá)

Use when the user works on **FIFA Skeletal Tracking Light 2026**, monocular broadcast pose estimation, `vaila_sam.py fifa`, or `fifa_skeletal_pipeline.py`.

---

## Challenge Overview

**FIFA Skeletal Tracking Light 2026** is a computer-vision challenge: given monocular broadcast video of FIFA World Cup matches, estimate the **3D pose of all visible players** in world coordinates.

- **Input:** single-camera broadcast video (HD), per-sequence camera parameters, pitch calibration points
- **Output:** per-frame 3D skeletons — **15 keypoints** (Body25/OpenPose subset) per person, in world space
- **Metric:** **MPJPE** (Mean Per Joint Position Error) in millimetres
- **Submission format:** ZIP containing `submission.npz` with per-sequence NumPy arrays of shape `(n_frames, n_persons, 15, 3)`

### External Resources

| Resource | URL |
|----------|-----|
| Challenge homepage | https://inside.fifa.com/innovation/innovation-programme/skeletal-tracking |
| Starter Kit (GitHub) | https://github.com/FIFA-Skeletal-Light-Tracking-Challenge/FIFA-Skeletal-Tracking-Starter-Kit-2026 |
| Dataset (Hugging Face) | https://huggingface.co/datasets/tijiang13/FIFA-Skeletal-Tracking-Light-2026 |
| Codabench — Validation | https://codabench.org/competitions/11681/ |
| Codabench — Test | https://www.codabench.org/competitions/11682/ |
| Kaggle Mirror | https://www.kaggle.com/competitions/fifa-skeletal-light/overview |
| WorldPose Paper | https://eth-ait.github.io/WorldPoseDataset/ |
| Discord (SoccerNet) | linked from starter kit README |

---

## Data Acquisition

### 1. Hugging Face Dataset

Accept access at [tijiang13/FIFA-Skeletal-Tracking-Light-2026](https://huggingface.co/datasets/tijiang13/FIFA-Skeletal-Tracking-Light-2026). This provides:

- `cameras/*.npz` — camera intrinsics + extrinsics per sequence
- `boxes/*.npy` — bounding boxes (from official pipeline)
- `skel_2d/*.npy` — estimated 2D keypoints (15 joints)
- `skel_3d/*.npy` — estimated 3D keypoints (15 joints)

### 2. Videos

Broadcasting videos come **separately from FIFA** (not on Hugging Face). Follow instructions in the starter kit or the WorldPose data request form.

### 3. Pitch Points

`pitch_points.txt` is vendored inside `vaila/fifa_starter_lib/pitch_points.txt`
(MIT, from the official starter kit). The `fifa bootstrap` subcommand copies
it into `<data_root>/pitch_points.txt` automatically.

### 4. Sequence Lists

- `sequences_full.txt` — all sequences
- `sequences_val.txt` — validation split (populate from the Codabench split!)
- `sequences_test.txt` — test split (no GT; submit to Codabench)

`fifa bootstrap` also generates these files from the available videos
(defaults: `val = full` with a warning, `test = empty`).

---

## Setup (step-by-step)

```bash
# 1. Switch to CUDA pyproject template (Linux workstation)
bash bin/use_pyproject_linux_cuda.sh
# Windows: pwsh bin/use_pyproject_win_cuda.ps1

# 2. Install FIFA + SAM + GPU extras
uv sync --extra gpu --extra fifa --extra sam

# 3. Clone sam_3d_body + download gated SAM 3D Body weights
#    Accept license on https://huggingface.co/facebook/sam-3d-body-dinov3 first,
#    then authenticate:
uv run hf auth login
bash bin/setup_fifa_sam3d.sh                # Linux / macOS
# pwsh bin/setup_fifa_sam3d.ps1            # Windows

# 4. (Optional) Download SAM 3 video weights for box generation via sam3
uv run vaila/vaila_sam.py --download-weights

# 5. Download challenge data
#    HF dataset: cameras, boxes, skel_2d, skel_3d
#    Videos: from FIFA (separate access)

# 6. Bootstrap the data layout (symlinks + sequences_*.txt + pitch_points.txt)
uv run vaila/vaila_sam.py fifa bootstrap \
  --videos-dir /path/to/FIFA_Challenge_2026_Video_Data/Videos \
  --data-root  /path/to/FIFA/data
```

### Vendored Code

- `vaila/fifa_starter_lib/` — **vendored MIT** `camera_tracker.py`, `postprocess.py`,
  and `pitch_points.txt` from the official starter kit. Updated via the steps in
  `vaila/fifa_starter_lib/VENDOR.md`.
- `sam_3d_body/` at repo root — cloned by `bin/setup_fifa_sam3d.sh` (NOT committed);
  installed into the uv environment with `uv pip install -e sam_3d_body/`.

---

## Data Layout

The pipeline expects this structure under a `data/` root directory:

```
data/
├── cameras/
│   ├── ARG_CRO_225412.npz
│   └── ...
├── images/
│   ├── ARG_CRO_225412/
│   │   ├── 00001.jpg
│   │   └── ...
│   └── ...
├── videos/
│   ├── ARG_CRO_225412.mp4
│   └── ...
├── boxes/
│   ├── ARG_CRO_225412.npy
│   └── ...
├── skel_2d/
│   ├── ARG_CRO_225412.npy
│   └── ...
├── skel_3d/
│   ├── ARG_CRO_225412.npy
│   └── ...
├── pitch_points.txt
├── sequences_full.txt
├── sequences_val.txt
└── sequences_test.txt
```

---

## Full Pipeline CLI

All FIFA subcommands are invoked through `vaila_sam.py fifa`:

### bootstrap — symlink videos + sequences_*.txt + pitch_points.txt

```bash
uv run vaila/vaila_sam.py fifa bootstrap \
  --videos-dir /data/FIFA/FIFA_Challenge_2026_Video_Data/Videos \
  --data-root  /data/FIFA/data \
  [--val-sequences /path/to/val.txt] \
  [--test-sequences /path/to/test.txt] \
  [--overwrite-pitch-points]
```

| Argument | Required | Description |
|----------|----------|-------------|
| `--videos-dir` | Yes | Source folder with raw `*.mp4` |
| `--data-root`  | Yes | Target FIFA data root |
| `--val-sequences` | No | Text file with Codabench val split |
| `--test-sequences` | No | Text file with Codabench test split |
| `--no-copy-fallback` | No | Fail when symlinks are denied instead of copying |
| `--overwrite-pitch-points` | No | Replace any existing `pitch_points.txt` |

Creates `data_root/videos/*.mp4` symlinks (copies on Windows without dev mode),
writes `sequences_full/val/test.txt`, and copies the vendored `pitch_points.txt`.
Run this **before `fifa prepare`** when the data root is empty.

### prepare — Videos to data/videos + data/images

```bash
uv run vaila/vaila_sam.py fifa prepare \
  --video-source DIR \
  --data-root data/ \
  [--sequences-out data/sequences_full.txt] \
  [--extract-fps 25.0]
```

| Argument | Required | Description |
|----------|----------|-------------|
| `--video-source` | Yes | Directory with source video files |
| `--data-root` | Yes | Target data directory |
| `--sequences-out` | No | Write a sequences list file |
| `--extract-fps` | No | FPS for frame extraction |

`prepare` does **not** create `cameras/`, `pitch_points.txt`, or official `boxes/` — those come from the starter kit / HF dataset.

### boxes — Generate bounding boxes

```bash
uv run vaila/vaila_sam.py fifa boxes \
  --data-root data/ \
  --sequences data/sequences_val.txt \
  [--source yolo] \
  [--yolo-model yolo11n.pt] \
  [--max-persons 25]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-root` | — | Data directory |
| `--sequences` | — | Sequences file |
| `--source` | `yolo` | `yolo` or `sam3` |
| `--yolo-model` | `yolo11n.pt` | YOLO model |
| `--sam3-checkpoint` | auto | SAM3 checkpoint for `--source sam3` |
| `--text-prompt` | `person` | Text prompt for SAM3 |
| `--max-persons` | `25` | Max persons per frame |

### preprocess — SAM 3D Body to skel_2d / skel_3d

```bash
uv run vaila/vaila_sam.py fifa preprocess \
  --data-root data/ \
  --sequences data/sequences_val.txt \
  [--no-skip]
```

Requires **CUDA**. Runs vendored SAM 3D Body to produce 2D and 3D skeleton estimates.

### baseline — Camera tracker + LBFGS to submission NPZ

```bash
uv run vaila/vaila_sam.py fifa baseline \
  --data-root data/ \
  --sequences data/sequences_full.txt \
  --output outputs/submission_full.npz \
  [--refine-interval 1] \
  [--export-camera] \
  [--calibration-dir outputs/calibration/]
```

Loads `pitch_points.txt`, per-sequence `cameras/*.npz`, `skel_*`, `boxes/*.npy`; runs `CameraTracker` + `process_sequence`; writes compressed NPZ.

### dlt-export — `cameras/*.npz` → per-frame `.dlt2d` / `.dlt3d` (vailá DLT)

After **`baseline --export-camera`** (or using starter-kit `cameras/*.npz`), export **one DLT row per frame** so **`vaila/rec2d.py`** and **`vaila/rec3d.py`** work on **moving** broadcast cameras. Fixed-tripod workflows stay on **`rec2d_one_dlt2d.py` / `rec3d_one_dlt3d.py`** (single DLT row).

```bash
uv run vaila/vaila_sam.py fifa dlt-export \
  --cameras-dir data/cameras \
  --output-dir outputs/dlt_per_frame \
  [--mode both|2d|3d] \
  [--undistort-pixels-dir path/to/sam_csvs]
# Or: uv run python -m vaila.fifa_to_dlt --input data/cameras --output outputs/dlt_per_frame
```

See **`vaila/help/vaila_sam.md`** section *Full broadcast pipeline (moving camera)*.

### pack — Split full NPZ to submission ZIP

```bash
# Validation
uv run vaila/vaila_sam.py fifa pack \
  --submission-full outputs/submission_full.npz \
  --data-root data/ \
  --output-dir outputs/ \
  --split val

# Test
uv run vaila/vaila_sam.py fifa pack \
  --submission-full outputs/submission_full.npz \
  --data-root data/ \
  --output-dir outputs/ \
  --split test
```

Reads `data/sequences_{val|test}.txt`, filters keys from the full NPZ, writes `submission_{split}.zip` containing `submission.npz`.

---

## End-to-End Example

```bash
# Step 1: Prepare
uv run vaila/vaila_sam.py fifa prepare \
  --video-source /path/to/raw_videos --data-root data/

# Step 2: Boxes
uv run vaila/vaila_sam.py fifa boxes \
  --data-root data/ --sequences data/sequences_val.txt

# Step 3: Preprocess (CUDA)
uv run vaila/vaila_sam.py fifa preprocess \
  --data-root data/ --sequences data/sequences_val.txt

# Step 4: Baseline
uv run vaila/vaila_sam.py fifa baseline \
  --data-root data/ --sequences data/sequences_full.txt \
  --output outputs/submission_full.npz

# Step 5: Pack
uv run vaila/vaila_sam.py fifa pack \
  --submission-full outputs/submission_full.npz \
  --data-root data/ --output-dir outputs/ --split val

uv run vaila/vaila_sam.py fifa pack \
  --submission-full outputs/submission_full.npz \
  --data-root data/ --output-dir outputs/ --split test
```

---

## Validation Workflow

1. Run the full pipeline to produce `outputs/submission_full.npz`
2. Pack with `--split val` to get `outputs/submission_val.zip`
3. Go to [Codabench Validation Portal](https://codabench.org/competitions/11681/)
4. Create an account / log in
5. Navigate to **My Submissions** and upload `submission_val.zip`
6. Wait for evaluation — MPJPE score is returned
7. Iterate: improve your model, re-run baseline, re-pack, re-submit

### Test Submission

Same process with `--split test` and the [Test Portal](https://www.codabench.org/competitions/11682/). Test ground truth is hidden.

---

## Code Map

| File | Role |
|------|------|
| `vaila/vaila_sam.py` | SAM 3 video segmentation + `fifa` CLI dispatch |
| `vaila/fifa_skeletal_pipeline.py` | FIFA pipeline orchestration (bootstrap/prepare/boxes/preprocess/baseline/dlt-export/pack) |
| `vaila/fifa_to_dlt.py` | FIFA `cameras/*.npz` (K,R,t,k) → per-frame `.dlt2d` / `.dlt3d` |
| `vaila/fifa_bootstrap.py` | `prepare_fifa_data_layout` — symlinks + sequences + pitch_points |
| `vaila/fifa_starter_lib/camera_tracker.py` | Vendored MIT camera tracker from starter kit |
| `vaila/fifa_starter_lib/postprocess.py` | Vendored MIT smoothing (`smoothen`) |
| `vaila/fifa_starter_lib/pitch_points.txt` | Vendored MIT FIFA pitch reference |
| `vaila/soccerfield_calib.py` | Companion DLT2D homography from 29 FIFA keypoints |
| `bin/setup_fifa_sam3d.sh` / `.ps1` | Clones sam_3d_body + downloads gated weights |
| `sam_3d_body/` | Cloned by the setup script (NOT committed) |
| `vaila/models/sam-3d-dinov3/` | SAM 3D Body weights (model.ckpt, mhr_model.pt) |
| `vaila/models/sam3/` | SAM 3 video weights (sam3.pt, sam3.1_multiplex.pt) |
| `tests/test_vaila_sam.py` | SAM helpers + GUI Help smoke test |
| `tests/test_fifa_skeletal_pipeline.py` | FIFA layout/packaging unit tests (no GPU) |
| `tests/test_fifa_bootstrap.py` | Bootstrap layout tests (symlinks + sequences) |
| `tests/test_soccerfield_calib.py` | DLT2D calibration smoke tests |
| `tests/test_fifa_to_dlt.py` | FIFA→DLT math + IO + undistort (needs `cv2`) |

---

## Known Issues

- **BFloat16 type mismatch:** SAM 3 can produce `Input type (c10::BFloat16) and bias type (float) should be the same` — vailá patches autocast and forces FP32 on the tracker backbone; edge cases may persist on some PyTorch builds
- **VRAM limits:** SAM 3 loads all frames onto GPU; use `--max-frames` or `SAM3_MAX_FRAMES` to cap
- **Gated HF repos:** Both `facebook/sam3` and `facebook/sam-3d-body-dinov3` require license acceptance; 403 means the account is not authorized — use `uv run hf auth login --force` with the correct account
- **Missing `fifa_starter_lib/`:** The MIT starter kit code is vendored at
  `vaila/fifa_starter_lib/`; update it with the steps in
  `vaila/fifa_starter_lib/VENDOR.md` if upstream changes.
- **Missing `sam_3d_body/`:** Run `bash bin/setup_fifa_sam3d.sh` (or the `.ps1`)
  to clone the Meta repo and `uv pip install -e` it.
- **No `cameras/` or `pitch_points.txt`:** `fifa bootstrap` seeds
  `pitch_points.txt` from the vendored copy; `cameras/*.npz` must come from the
  HF dataset or from `vaila/soccerfield_calib.py` (companion tool — drops
  `cameras/<stem>_homography.npz` when `--data-root` is passed).

---

## Evaluation

- **Primary metric:** MPJPE (Mean Per Joint Position Error) in millimetres
- **Per-sequence NPZ arrays:** shape `(n_frames, n_persons, 15, 3)`
- **15 keypoints** (OpenPose Body25 subset): indices `[0, 2, 5, 3, 6, 4, 7, 9, 12, 10, 13, 11, 14, 22, 19]`
- Evaluation is automated on Codabench after ZIP upload

---

## Tests

```bash
# Fast unit tests (no GPU)
uv run pytest tests/test_fifa_skeletal_pipeline.py -v

# SAM helpers (no GPU)
uv run pytest tests/test_vaila_sam.py -v

# GPU smoke test
export VAILA_TEST_SAM_GPU=1
uv run pytest tests/test_vaila_sam.py -v -k sam3_smoke
```

See [AGENTS.md](../../../AGENTS.md) for the canonical agent summary.
