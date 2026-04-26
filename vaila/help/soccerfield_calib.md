# soccerfield_calib

## Module Information

- **Category:** Multimodal Analysis / Sports Field Calibration
- **File:** `vaila/soccerfield_calib.py`
- **Version:** 0.1.0 (April 2026)
- **Author:** Paulo Santiago — paulosantiago@usp.br
- **GUI Interface:** Yes — button **Soccer-Field Calib** in Frame B
- **CLI Interface:** Yes
- **License:** AGPL-3.0

---

## Description

`soccerfield_calib.py` fits a **DLT2D homography** (8 coefficients) that
maps a fixed broadcast frame to the **FIFA 105 × 68 m soccer field**,
using a small set of clicked pixel keypoints (≥ 6) and the canonical
3D reference [`vaila/models/soccerfield_ref3d.csv`](../models/soccerfield_ref3d.csv)
(29 named points, Z = 0 ground plane).

Outputs:

- `<stem>_ref2d.csv` — world XY pairs used for fitting
- `<stem>.dlt2d` — 8 DLT coefficients (compatible with `vaila/rec2d.py`)
- `<stem>_homography_report.txt` — per-point reprojection error
- `cameras/<stem>_homography.npz` (when `--data-root` is given) —
  fallback for FIFA sequences without an official `cameras/<stem>.npz`

`# TODO: Z vertical (DLT3D future work)` — only Z = 0 ground plane today.

---

## When to use which calibration

| Tool | Camera | Per frame? | Use |
|---|---|---|---|
| `soccerfield_calib.py` | **fixed / static frame** | **No** (single homography) | Static cam, demo, fallback when there is no FIFA NPZ |
| `fifa_to_dlt.py` (a.k.a. **`fifa dlt-export`**) | **moving broadcast** | **Yes** (one row / frame) | Pan/tilt/zoom — required for real broadcast |
| `rec2d_one_dlt2d.py` | fixed | one row of 8 coeffs | Tripod 2D reconstruction |
| `rec3d_one_dlt3d.py` | fixed (multi-cam) | one row of 11 coeffs/cam | Static lab |
| `rec2d.py` / `rec3d.py` | moving | per-frame DLT | Broadcast |

> If the camera moves, **always** use the FIFA `cameras/*.npz` route.
> `soccerfield_calib.py` is a single-frame / static-camera tool.

---

## Step-by-step (GUI)

### Step 1 — Open

Click **Soccer-Field Calib** in vailá Frame B.

### Step 2 — Pick the broadcast video / fixed frame

Choose an MP4 / AVI / MOV. The dialog opens `getpixelvideo.py` so you
can click the field keypoints on the chosen frame.

### Step 3 — Click the keypoints

Click **at least 6** of the named soccer-field points. Use the names
listed in [`vaila/models/soccerfield_ref3d.csv`](../models/soccerfield_ref3d.csv)
(corners, midfield_left/right, center_field, left_penalty_spot,
right_penalty_spot, penalty arc tops, …). The defaults suggested by
the GUI prioritise the points that are easiest to see in zoomed
broadcast crops.

### Step 4 — Save

The script fits the homography, prints a per-point error report and
writes the 4 output files. If you set **FIFA data-root**, it also
drops `cameras/<stem>_homography.npz`.

---

## CLI quick recipes

### Recipe 1 — Static frame, GUI for clicking

```bash
uv run vaila/soccerfield_calib.py \
  -v /path/to/video.mp4 \
  -o /path/to/output_dir
```

### Recipe 2 — Use a pre-clicked pixel CSV (skip GUI)

```bash
uv run vaila/soccerfield_calib.py \
  -v /path/to/video.mp4 \
  -p /path/to/pixels_clicked.csv \
  --frame 0 \
  -o /path/to/output_dir
```

### Recipe 3 — FIFA fallback (no official cameras NPZ)

```bash
uv run vaila/soccerfield_calib.py \
  -v /path/to/data/videos/SEQ.mp4 \
  --data-root /path/to/data \
  -o /path/to/data/cameras
```

This produces `cameras/SEQ_homography.npz` that the FIFA pipeline can
read as a fallback.

### Recipe 4 — Restrict keypoint set

```bash
uv run vaila/soccerfield_calib.py \
  -v video.mp4 -o out/ \
  --keypoints bottom_left_corner,bottom_right_corner,top_left_corner,top_right_corner,center_field,left_penalty_spot
```

### Recipe 5 — From a SAM3 batch result

```bash
uv run vaila/soccerfield_calib.py \
  --from-sam /path/to/processed_sam_<ts>/<video>/ \
  --pixels /path/to/clicks.csv
```

`--from-sam` defaults `--video` to the SAM3 overlay MP4 and `--output`
to `<sam_dir>/calib/`.

### Recipe 6 — List valid keypoint names

```bash
uv run vaila/soccerfield_calib.py --list-keypoints
```

---

## CLI flags

| Flag | Purpose |
|---|---|
| `-v, --video` | Input video (triggers GUI clicking) |
| `-p, --pixels` | Pre-picked pixel CSV (skip GUI) |
| `-r, --ref3d` | FIFA reference CSV (default `models/soccerfield_ref3d.csv`) |
| `-o, --output` | Output directory |
| `--frame` | Frame index for paired-column CSVs |
| `--data-root` | FIFA data root → also writes `cameras/<stem>_homography.npz` |
| `--keypoints` | Comma-separated list overriding GUI suggestions |
| `--list-keypoints` | Print valid kp names and exit |
| `--from-sam` | Use a SAM3 per-video output directory as input |

---

## Pixel CSV format

Two flavours are accepted:

1. **Long** — `name,x,y` (or `kp_name,x,y`). Names must match the FIFA reference CSV.
2. **Wide** — `frame,p1_x,p1_y,p2_x,p2_y,…` (the `getpixelvideo`
   layout). Combine with `--frame N` to pick a row.

Both are produced by `vaila.soccerfield_keypoints_ai` (the wide layout
is `field_keypoints_getpixelvideo.csv`).

---

## Output files

```
<output>/
  <stem>_ref2d.csv                  # world XY for fitting
  <stem>.dlt2d                      # 8 DLT coefficients (1 row)
  <stem>_homography_report.txt      # per-point error (px, m)
  cameras/<stem>_homography.npz     # only when --data-root is set
```

`<stem>` is derived from the video filename or the pixel CSV.

---

## Reading the error report

The report lists for each used keypoint:

- pixel residual `(px, py)` — reprojection from world → pixel
- world residual `(mx, my)` — projection error in metres after
  applying the inverse homography

Typical broadcast values:

| Quality | Pixel error | World error |
|---|---|---|
| Excellent | < 2 px | < 0.30 m |
| Acceptable | < 5 px | < 0.80 m |
| Re-click | > 8 px | > 1.5 m |

---

## End-to-end recipe (with the AI seed)

```bash
# 1. AI seed (32 keypoints) — Field KPs (AI)
uv run python -m vaila.soccerfield_keypoints_ai \
  --mode video -i video.mp4 -o out_kps/ \
  --backend ultralytics \
  --weights vaila/models/runs/pose_fifa/pitch32_recipeA_400ep/weights/best.pt \
  --imgsz 1280 --conf 0.30 --draw-min-conf 0.40 --device 0 \
  --stride 1 --overlay-video

# 2. Optional manual refine in getpixelvideo (open the wide CSV).

# 3. DLT2D homography (single static frame from the video)
uv run vaila/soccerfield_calib.py \
  -v video.mp4 \
  -p out_kps/processed_field_kps_*/field_keypoints_getpixelvideo.csv \
  --frame 0 -o out_calib/

# 4. Reconstruct player pixels → field metres
uv run vaila/rec2d.py \
  --dlt-file out_calib/video.dlt2d \
  --input-dir player_pixels/ \
  --output-dir player_world/ --rate 30
```

For **broadcast (moving camera)**, replace step 3 by the FIFA
`cameras/*.npz` → per-frame DLT route — see
`vaila/help/vaila_sam.html` (section *Full broadcast pipeline*).

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| "Need at least 6 points" | Not enough clicks | Click 8–10 well-distributed kps |
| Huge world error (>5 m) | Wrong correspondence | Verify name ↔ world xy in `--list-keypoints` |
| Fit fine, players still wrong | Camera moves between frames | Use `fifa_to_dlt.py` (per-frame DLT) instead |
| `pitch_keypoints.png` index ≠ FIFA name | Two indexing systems | The 32-id system (AI seed) is generic; the 29 FIFA names are the calibration targets — match by visual location |

---

## Related

- `soccerfield_keypoints_ai.py` — AI seed for the 32 keypoints
- `getpixelvideo.py` — manual click / refine
- `fifa_to_dlt.py` — per-frame DLT for moving camera
- `vaila/rec2d.py`, `vaila/rec3d.py` — pixel → world reconstruction
- `vaila/dlt2d.py`, `vaila/dlt3d.py` — DLT math
- `vaila/help/vaila_sam.html` — FIFA pipeline + broadcast section

Generated: April 26, 2026.
