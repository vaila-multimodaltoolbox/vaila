# soccerfield_keypoints_ai

## Module Information

- **Category:** Multimodal Analysis / Sports Field Calibration
- **File:** `vaila/soccerfield_keypoints_ai.py`
- **Version:** 0.1.0 (April 2026)
- **Author:** Paulo Santiago — paulosantiago@usp.br
- **GUI Interface:** Yes (Tkinter) — button **Field KPs (AI)** in Frame B
- **CLI Interface:** Yes
- **License:** AGPL-3.0

---

## Description

**`soccerfield_keypoints_ai`** detects up to **32 soccer-field
keypoints** (corners, penalty area corners, centre circle, penalty
spots, …) in **broadcast video** using a YOLO-pose model. The keypoints
become the input for `soccerfield_calib.py` and for FIFA per-frame DLT
export (`vaila/fifa_to_dlt.py`).

### Two backends

| Backend | When to use | Requires |
|---|---|---|
| `ultralytics` | Local `.pt` weights | YOLO-pose `.pt` file |
| `roboflow` | Cloud inference, no local weights | `inference`, `supervision`, `ROBOFLOW_API_KEY` |

### Two modes

| Mode | Output | Typical use |
|---|---|---|
| `frame` | one frame, raw + template CSV + overlay PNG | sanity check |
| `video` | per-frame CSV (long + wide getpixelvideo) + overlay MP4 | full pipeline |

---

## Default model (ships with vailá, recipe-A 400 epochs)

Path: `vaila/models/runs/pose_fifa/pitch32_recipeA_400ep/weights/best.pt`

| Metric | Value |
|---|---|
| Box mAP50 | 0.960 |
| Box mAP50-95 | 0.788 |
| **Pose mAP50** | **0.945** |
| **Pose mAP50-95** | **0.813** |

Trained on the public `roboflow/football-field-detection-f07vi` dataset
(255 train + 34 val) using `yolo26s-pose.pt` with the **recipe that
escapes pose-collapse**:

```
imgsz=1280 batch=8 mosaic=0 mixup=0 close_mosaic=0 erasing=0
pose=25 kobj=2 device=0 patience=80
```

---

## Step-by-step (most users)

### Step 1 — Open the GUI button or run the CLI

Either:

- Click **Field KPs (AI)** in the vailá main window (Frame B, bottom row), **or**
- Run `uv run python -m vaila.soccerfield_keypoints_ai` (no args → opens GUI)

### Step 2 — Pick the broadcast video and an output folder

The GUI asks for:

1. Soccer video (`.mp4 .avi .mov .mkv .webm`)
2. Output folder (vailá creates `processed_field_kps_<timestamp>/` inside)

### Step 3 — Configure the dialog

| Field | Meaning | Recommended |
|---|---|---|
| Mode | `frame` or `video` | `video` for FIFA pipeline |
| Backend | `ultralytics` or `roboflow` | `ultralytics` (default model bundled) |
| Stride | every Nth frame | `1` for full clip, `5` for smoke test |
| Max frames | cap on processed frames | empty for full clip |
| Confidence | YOLO `conf` threshold | `0.30` |
| Draw/save min conf | overlay filter | `0.40` |
| imgsz | inference size | `1280` |
| Device | GPU id or empty | `0` for CUDA, empty for auto |
| Ultralytics weights | path to `.pt` | leave empty → bundled `pitch32_recipeA_400ep/best.pt` |

For Roboflow: fill `Roboflow model id` and `Roboflow API key` (or
export `ROBOFLOW_API_KEY` before launching).

### Step 4 — Click **Run**

A success messagebox shows the output folder.

---

## CLI quick recipes

### Recipe 1 — Smoke test (default model, 60 frames)

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

### Recipe 2 — Full clip (every frame)

```bash
uv run python -m vaila.soccerfield_keypoints_ai \
  --mode video \
  -i path/to/video.mp4 \
  -o path/to/output \
  --backend ultralytics \
  --weights vaila/models/runs/pose_fifa/pitch32_recipeA_400ep/weights/best.pt \
  --imgsz 1280 --conf 0.30 --draw-min-conf 0.40 \
  --device 0 --stride 1 --overlay-video
```

### Recipe 3 — Batch over a folder of videos

```bash
for f in /path/to/videos/*.mp4; do
  uv run python -m vaila.soccerfield_keypoints_ai \
    --mode video -i "$f" \
    -o /path/to/outputs_pitch_kps_full \
    --backend ultralytics \
    --weights vaila/models/runs/pose_fifa/pitch32_recipeA_400ep/weights/best.pt \
    --imgsz 1280 --conf 0.30 --draw-min-conf 0.40 \
    --device 0 --stride 1 --overlay-video
done
```

### Recipe 4 — Roboflow backend (no local model)

```bash
export ROBOFLOW_API_KEY="YOUR_KEY"
uv run python -m vaila.soccerfield_keypoints_ai \
  --mode video \
  -i path/to/video.mp4 \
  -o path/to/output \
  --backend roboflow \
  --roboflow-model-id football-field-detection-f07vi/14 \
  --conf 0.3 --draw-min-conf 0.05 \
  --stride 1 --overlay-video
```

### Recipe 5 — Single-frame sanity check (frame mode)

```bash
uv run python -m vaila.soccerfield_keypoints_ai \
  --mode frame -i video.mp4 -o out/ \
  --backend ultralytics \
  --weights vaila/models/runs/pose_fifa/pitch32_recipeA_400ep/weights/best.pt \
  --frame 0 --imgsz 1280 --conf 0.20 --draw-min-conf 0.20
```

### Recipe 6 — CPU fallback

```bash
CUDA_VISIBLE_DEVICES="" uv run python -m vaila.soccerfield_keypoints_ai \
  --mode video -i video.mp4 -o out/ \
  --backend ultralytics \
  --weights vaila/models/runs/pose_fifa/pitch32_recipeA_400ep/weights/best.pt \
  --device cpu --stride 30 --max-frames 5 --overlay-video
```

---

## CLI flags

| Flag | Type | Default | Notes |
|---|---|---|---|
| `-i, --input` | path | — | Input video |
| `-o, --output` | path | — | Output base folder |
| `--mode` | `frame`/`video` | `frame` | Mode |
| `--frame` | int | `0` | Frame index (frame mode) |
| `--start` | int | `0` | First frame (video mode) |
| `--stride` | int | `10` | Process every Nth frame |
| `--max-frames` | int | full clip | Cap (video mode) |
| `--overlay-video` | flag | off | Write overlay MP4 |
| `--backend` | `roboflow`/`ultralytics` | `roboflow` | Backend |
| `--conf` | float | `0.3` | Detection confidence |
| `--draw-min-conf` | float | `0.3` | Overlay/save filter |
| `--imgsz` | int | `1280` | Ultralytics inference size |
| `--device` | str/int | auto | `0`, `cpu`, `mps` … |
| `--roboflow-model-id` | str | `football-field-detection-f07vi/14` | Roboflow model |
| `--weights` | path | — | Local YOLO `.pt` (Ultralytics backend) |

---

## Outputs

```
processed_field_kps_<timestamp>/
  field_keypoints_video.csv               # long: frame, name, x, y, conf
  field_keypoints_getpixelvideo.csv       # wide: frame, p1_x, p1_y, …, p32_x, p32_y
  field_keypoints_overlay_markers.csv     # same wide layout, filtered by --draw-min-conf
  field_keypoints_overlay.mp4             # only with --overlay-video
  README_field_keypoints_video.txt        # run metadata
```

The wide CSV is **directly compatible with `getpixelvideo.py`**: open
it there and refine clicks if needed.

### Keypoint id ↔ semantic name

The detector outputs generic `kp_00 … kp_31`. The semantic mapping
(centre circle, penalty spots, …) follows
[`vaila/models/hf_datasets/football-pitch-detection/pitch_keypoints.png`](../models/hf_datasets/football-pitch-detection/pitch_keypoints.png)
and the FIFA reference layout
[`vaila/models/soccerfield_ref3d_fifa.csv`](../models/soccerfield_ref3d_fifa.csv).

---

## Training a new detector (advanced)

```bash
uv run yolo pose train \
  model=yolo26s-pose.pt \
  data=vaila/models/hf_datasets/football-pitch-detection/data/data.yaml \
  epochs=400 imgsz=1280 batch=8 \
  mosaic=0.0 mixup=0.0 close_mosaic=0 erasing=0.0 \
  pose=25.0 kobj=2.0 device=0 patience=80 \
  project=vaila/models/runs/pose_fifa name=pitch32_recipeA_400ep
```

Important: **do not enable `mosaic` or `mixup`** — they collapse the
32 keypoints onto a single point even while box mAP stays high
(documented in
`.claude/skills/soccer-field-keypoints-yolo/SKILL.md`).

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| All 32 keypoints in a small cluster | Old/collapsed model | Use the bundled `pitch32_recipeA_400ep/best.pt` or retrain |
| `cv2.VideoCapture not found` | Broken OpenCV install | `uv pip install --reinstall opencv-python==4.10.0.84` |
| `Roboflow backend requires inference` | Missing optional deps | `uv add inference supervision` and set `ROBOFLOW_API_KEY` |
| GPU OOM | imgsz too high or other job using GPU | `--imgsz 960` or `--device cpu` |
| Keypoints ok in centre, weak at borders | Strong perspective | normal — confidence drops correctly outside the field |

---

## Related

- `vaila/soccerfield_calib.py` — DLT2D homography from the same kp set
- `vaila/fifa_to_dlt.py` — FIFA `cameras/*.npz` → per-frame DLT
- `vaila/getpixelvideo.py` — manual click + refine of the wide CSV
- `.claude/skills/soccer-field-keypoints-yolo/SKILL.md` — full training recipe
- `.claude/skills/fifa-vaila-continuation/SKILL.md` — FIFA challenge resume

Generated: April 26, 2026.
