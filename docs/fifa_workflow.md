# FIFA Skeletal Tracking Light — vailá Workflow (CLI + GUI)

> Step-by-step end-to-end recipe to win the
> [FIFA Skeletal Tracking Light 2026](https://inside.fifa.com/innovation/innovation-programme/skeletal-tracking)
> challenge with vailá.
>
> **Date:** April 2026 · **License:** AGPL-3.0
>
> Companion help pages:
> - [`vaila_sam.html`](../vaila/help/vaila_sam.html) — SAM 3 video + FIFA pipeline
> - [`soccerfield_keypoints_ai.html`](../vaila/help/soccerfield_keypoints_ai.html) — AI seed for the 32 pitch keypoints
> - [`soccerfield_calib.html`](../vaila/help/soccerfield_calib.html) — DLT2D homography from clicked keypoints
> - [`sports_fields_courts.html`](../vaila/help/sports_fields_courts.html) — Draw the FIFA reference field
> - [`getpixelvideo.html`](../vaila/help/getpixelvideo.html) — Manual click / refine

---

## 1. The five-stage pipeline at a glance

```
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ 0. Environment  │───▶│ 1. Player masks  │───▶│ 2. Pitch         │
│    (uv, CUDA,   │    │    SAM 3 video   │    │    keypoints     │
│     models)     │    │    (vaila_sam.py)│    │ (soccerfield_kps_│
│                 │    │                  │    │       ai.py)     │
└─────────────────┘    └──────────────────┘    └──────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ 5. Submit       │◀───│ 4. 3D pose +     │◀───│ 3. Camera        │
│   submission.zip│    │    DLT export    │    │   calibration    │
│   to Codabench  │    │ (fifa_skeletal_  │    │ (soccerfield_    │
│                 │    │   pipeline.py +  │    │  calib.py /      │
│                 │    │  fifa_to_dlt.py) │    │  fifa cameras)   │
└─────────────────┘    └──────────────────┘    └──────────────────┘
```

| Stage | vailá module | GUI button | CLI script |
|---|---|---|---|
| 0 | — | — | `bin/use_pyproject_linux_cuda.sh`; `uv sync --extra gpu --extra sam --extra fifa` |
| 1 | `vaila_sam.py` | **vaila_sam** | `uv run vaila/vaila_sam.py` |
| 2 | `soccerfield_keypoints_ai.py` | **Field KPs (AI)** | `uv run python -m vaila.soccerfield_keypoints_ai` |
| 3 | `soccerfield_calib.py` (static) **or** `fifa cameras` (broadcast) | **Soccer-Field Calib**, **FIFA cams→DLT** | `uv run vaila/soccerfield_calib.py`, `uv run vaila/vaila_sam.py fifa baseline --export-camera` |
| 4 | `fifa_skeletal_pipeline.py`, `fifa_to_dlt.py` | **FIFA cams→DLT** | `uv run vaila/vaila_sam.py fifa baseline / dlt-export / pack` |
| 5 | — | — | upload `outputs/submission_*.zip` |

---

## 2. Stage 0 — Environment (one-time, ≤ 10 min)

> A workstation with NVIDIA CUDA is required. SAM 3 video and SAM 3D Body
> have **no** CPU / Apple-Metal fallback (`AGENTS.md`).

### 2.1 Switch to the CUDA template

```bash
cd /path/to/vaila
bash bin/use_pyproject_linux_cuda.sh   # Linux  (or pwsh bin/use_pyproject_win_cuda.ps1)
```

### 2.2 Sync extras

```bash
uv sync --extra gpu --extra sam --extra fifa
```

### 2.3 Hugging Face access

Accept the licence on the gated repos, then login:

```bash
uv run hf auth login           # paste a HF token with read access
```

Repos to accept:

- [facebook/sam3](https://huggingface.co/facebook/sam3) — SAM 3 video weights
- [facebook/sam-3d-body-dinov3](https://huggingface.co/facebook/sam-3d-body-dinov3) — SAM 3D Body
- [tijiang13/FIFA-Skeletal-Tracking-Light-2026](https://huggingface.co/datasets/tijiang13/FIFA-Skeletal-Tracking-Light-2026) — challenge dataset

### 2.4 Download weights / clone SAM 3D Body

```bash
# SAM 3 video weights (auto-installs to vaila/models/sam3/sam3.pt)
uv run vaila/vaila_sam.py --download-weights

# SAM 3D Body — clones sam_3d_body/ + downloads gated DINOv3 weights
bash bin/setup_fifa_sam3d.sh        # Linux/macOS
# pwsh bin/setup_fifa_sam3d.ps1     # Windows
```

### 2.5 Smoke check

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
uv run pytest tests/test_vaila_sam.py tests/test_fifa_skeletal_pipeline.py -v
```

---

## 3. Stage 1 — Player masks (SAM 3 video)

### 3.1 GUI

1. `uv run vaila.py`
2. In Frame B click **vaila_sam** → choose video / folder / output
3. Set **Text prompt** = `player`, tick **Save overlay MP4**, click **Run**

### 3.2 CLI (single video)

```bash
uv run vaila/vaila_sam.py \
  -i tests/sport_fields/ENG_FRA_220243.mp4 \
  -o tests/sport_fields/runs/sam3 \
  -t player \
  --max-frames 80 --max-input-long-edge 1280 \
  --postprocess-points foot
```

### 3.3 CLI (batch — folder)

```bash
uv run vaila/vaila_sam.py \
  -i /data/FIFA/.../Videos \
  -o /data/FIFA/runs/sam3_full \
  -t player --postprocess-points foot
```

> Batch uses one subprocess per video (auto), so CUDA memory resets
> between clips. Disable with `--no-isolate-batch` only for debugging.

### 3.4 Outputs

```
processed_sam_<timestamp>/<video_stem>/
  masks/                       # PNG per frame (unless --no-png)
  <stem>_sam_overlay.mp4       # coloured masks (unless --no-overlay)
  sam_frames_meta.csv
  sam_points.csv               # only with --postprocess-points
  README_sam.txt
```

The `sam_points.csv` is the per-frame foot/centroid CSV that feeds
`rec2d.py` after calibration.

Help: [`vaila_sam.html`](../vaila/help/vaila_sam.html).

---

## 4. Stage 2 — Pitch keypoints (Field KPs AI)

### 4.1 GUI

1. In Frame B click **Field KPs (AI)**
2. Pick the broadcast video and an output folder
3. Set Mode = `video`, Backend = `ultralytics`, Stride = `1` (or `5` for smoke),
   Confidence = `0.30`, Draw min conf = `0.40`, imgsz = `1280`, Device = `0`
4. Leave **Ultralytics weights** empty → uses bundled
   `vaila/models/runs/pose_fifa/pitch32_recipeA_400ep/weights/best.pt`
   (Pose mAP50 = 0.945)
5. Tick **Write overlay .mp4**, click **Run**

### 4.2 CLI (smoke test, 60 frames)

```bash
uv run python -m vaila.soccerfield_keypoints_ai \
  --mode video \
  -i tests/sport_fields/ENG_FRA_220243.mp4 \
  -o tests/sport_fields/runs/pitch_kps \
  --backend ultralytics \
  --weights vaila/models/runs/pose_fifa/pitch32_recipeA_400ep/weights/best.pt \
  --imgsz 1280 --conf 0.30 --draw-min-conf 0.40 \
  --device 0 --stride 5 --max-frames 60 --overlay-video
```

### 4.3 CLI (full clip)

Drop `--max-frames` and use `--stride 1`.

### 4.4 Outputs

```
processed_field_kps_<timestamp>/
  field_keypoints_video.csv               # long
  field_keypoints_getpixelvideo.csv       # wide (for getpixelvideo / soccerfield_calib)
  field_keypoints_overlay_markers.csv
  field_keypoints_overlay.mp4
  README_field_keypoints_video.txt
```

Help: [`soccerfield_keypoints_ai.html`](../vaila/help/soccerfield_keypoints_ai.html).

---

## 5. Stage 3 — Camera calibration

### 5.1 Static frame / fallback (single homography)

GUI: **Soccer-Field Calib** — clicks 6+ keypoints in `getpixelvideo`.

CLI:

```bash
uv run vaila/soccerfield_calib.py \
  -v video.mp4 \
  -p processed_field_kps_*/field_keypoints_getpixelvideo.csv \
  --frame 0 \
  -o out_calib/
```

Output: `video.dlt2d` (8 coefficients, one row).

Help: [`soccerfield_calib.html`](../vaila/help/soccerfield_calib.html).

### 5.2 Broadcast / moving camera (per-frame DLT)

This is the **real** challenge case. You need one DLT row per video frame.

#### 5.2.1 FIFA bootstrap

```bash
uv run vaila/vaila_sam.py fifa bootstrap \
  --videos-dir /data/FIFA/.../Videos \
  --data-root  /data/FIFA/data
```

This creates the layout:

```
data/
  videos/<seq>.mp4         (symlinks or copies)
  cameras/<seq>.npz        (initial K, R, t, k from starter kit)
  pitch_points.txt
  sequences_full.txt / _val.txt / _test.txt
```

#### 5.2.2 Refresh per-frame cameras (baseline updates them)

```bash
uv run vaila/vaila_sam.py fifa baseline \
  --data-root data/ \
  --sequences data/sequences_full.txt \
  --output    outputs/submission_full.npz \
  --export-camera \
  --calibration-dir data/cameras
```

#### 5.2.3 Convert to vailá DLT (2D + 3D)

```bash
uv run vaila/vaila_sam.py fifa dlt-export \
  --cameras-dir data/cameras \
  --output-dir  outputs/dlt_per_frame \
  --mode both
# or via the GUI button: FIFA cams→DLT
```

Output: `<seq>.dlt2d` and `<seq>.dlt3d`, each with **N rows** (one per frame).

---

## 6. Stage 4 — 3D pose + reconstruction

### 6.1 Generate 2D + 3D skeletons (SAM 3D Body)

```bash
# Boxes (YOLO or SAM3)
uv run vaila/vaila_sam.py fifa boxes \
  --data-root data/ --sequences data/sequences_val.txt

# 2D + 3D skeletons (CUDA required)
uv run vaila/vaila_sam.py fifa preprocess \
  --data-root data/ --sequences data/sequences_val.txt
```

### 6.2 Run baseline → submission NPZ

```bash
uv run vaila/vaila_sam.py fifa baseline \
  --data-root data/ \
  --sequences data/sequences_full.txt \
  --output    outputs/submission_full.npz \
  --export-camera
```

### 6.3 (optional) Project player pixels to field metres

For 2D field plane:

```bash
uv run vaila/rec2d.py \
  --dlt-file outputs/dlt_per_frame/<seq>.dlt2d \
  --input-dir player_pixels/ \
  --output-dir player_world/ \
  --rate 30
```

For multi-camera 3D (each camera's `.dlt3d` has N rows, time-aligned):

```bash
uv run vaila/rec3d.py \
  --dlt-files outputs/dlt_per_frame/camA.dlt3d outputs/dlt_per_frame/camB.dlt3d \
  --input-dir merged_pixels/ \
  --output-dir rec3d_out/ \
  --rate 30
```

> Static lab cams only? Use `rec2d_one_dlt2d.py` / `rec3d_one_dlt3d.py`
> instead (single-row DLT).

---

## 7. Stage 5 — Submit

```bash
# Validation
uv run vaila/vaila_sam.py fifa pack \
  --submission-full outputs/submission_full.npz \
  --data-root data/ --output-dir outputs/ --split val

# Test
uv run vaila/vaila_sam.py fifa pack \
  --submission-full outputs/submission_full.npz \
  --data-root data/ --output-dir outputs/ --split test
```

Upload:

- `outputs/submission_val.zip` → [Codabench validation](https://codabench.org/competitions/11681/)
- `outputs/submission_test.zip` → [Codabench test](https://www.codabench.org/competitions/11682/)

Primary metric: **MPJPE** (Mean Per Joint Position Error, mm).

---

## 8. Smoke test on a single video (most useful one-liner)

```bash
# 1) SAM 3 — players
uv run vaila/vaila_sam.py \
  -i tests/sport_fields/ENG_FRA_220243.mp4 \
  -o tests/sport_fields/runs/sam3 \
  -t player --max-frames 80 --postprocess-points foot

# 2) Pitch keypoints
uv run python -m vaila.soccerfield_keypoints_ai \
  --mode video \
  -i tests/sport_fields/ENG_FRA_220243.mp4 \
  -o tests/sport_fields/runs/pitch_kps \
  --backend ultralytics \
  --weights vaila/models/runs/pose_fifa/pitch32_recipeA_400ep/weights/best.pt \
  --imgsz 1280 --conf 0.30 --draw-min-conf 0.40 \
  --device 0 --stride 5 --max-frames 60 --overlay-video

# 3) DLT2D homography (single frame as a sanity check)
uv run vaila/soccerfield_calib.py \
  -v tests/sport_fields/ENG_FRA_220243.mp4 \
  -p tests/sport_fields/runs/pitch_kps/processed_field_kps_*/field_keypoints_getpixelvideo.csv \
  --frame 0 \
  -o tests/sport_fields/runs/calib

# 4) Open the overlays and the report
xdg-open tests/sport_fields/runs/sam3/processed_sam_*/*/*_sam_overlay.mp4
xdg-open tests/sport_fields/runs/pitch_kps/processed_field_kps_*/field_keypoints_overlay.mp4
```

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `torch.cuda.is_available() == False` | wrong template | run `bin/use_pyproject_linux_cuda.sh` then `uv sync --extra gpu` |
| `403 from huggingface.co` | licence not accepted | accept on HF, then `uv run hf auth login` |
| SAM 3 OOM in batch | 1 leak in CUDA workspaces | already mitigated by subprocess-per-video; do not pass `--no-isolate-batch` |
| Pitch keypoints all clustered | old / collapsed model | use the bundled `pitch32_recipeA_400ep/best.pt` |
| Players ok, world coords wrong | camera moves but you used `soccerfield_calib` | switch to `fifa dlt-export` (per-frame DLT) |
| `cv2.VideoCapture not found` | broken OpenCV | `uv pip install --reinstall opencv-python==4.10.0.84` |
| `hf` ModuleNotFoundError typer | env issue | `uv add typer` or set `HF_TOKEN` in `.env` |

---

## 10. Where to read more

- **In the app:** click **Help** in any of the dialogs (`vaila_sam`,
  `Field KPs (AI)`, `Soccer-Field Calib`).
- **Per-module help:** `vaila/help/<module>.html` (auto-opens from the
  GUI Help button).
- **Skills (for AI agents):**
  - `.claude/skills/sam3-video/SKILL.md`
  - `.claude/skills/soccer-field-keypoints-yolo/SKILL.md`
  - `.claude/skills/fifa-skeletal-tracking/SKILL.md`
  - `.claude/skills/fifa-vaila-continuation/SKILL.md`
- **Repository docs:** `AGENTS.md`, `CLAUDE.md` and `.cursor/rules/vaila.mdc`.

---

Generated April 2026 — vailá Multimodal Toolbox.
