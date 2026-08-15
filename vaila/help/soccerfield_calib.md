# soccerfield_calib

## Module Information

- **Category:** Multimodal Analysis / Sports Field Calibration
- **File:** `vaila/soccerfield_calib.py`
- **Version:** 0.3.104 (11 August 2026)
- **Author:** Paulo Santiago — paulosantiago@usp.br
- **GUI Interface:** Yes — **Frame B → Soccer Tools → Soccer-Field Calib**
- **CLI Interface:** Yes
- **License:** AGPL-3.0

---

## Description

`soccerfield_calib.py` fits a **DLT2D homography** (8 coefficients) that
maps a fixed broadcast frame to the **FIFA 105 × 68 m soccer field**,
using a small set of clicked pixel keypoints (≥ 6) and the canonical
3D reference:

- **Legacy 29-pt metre grid:** [`vaila/models/soccerfield_ref3d.csv`](../models/soccerfield_ref3d.csv) (getpixelvideo manual clicks with point names).

The automatic 32-point `p1`…`p32` sequence is consumed directly by
`soccerfield_vitruvian_dlt3d.py`. For this legacy named-point module, columns
must carry the semantic names in the selected reference or be supplied in
matching order through `--keypoints`.

Outputs:

- `<stem>_ref2d.csv` — world XY pairs used for fitting
- `<stem>.dlt2d` — one selected frame with 8 DLT coefficients (compatible with `vaila/rec2d.py`)
- `<stem>_homography_report.txt` — per-point reprojection error
- `cameras/<stem>_homography.npz` (when `--data-root` is given) —
  fallback for FIFA sequences without an official `cameras/<stem>.npz`

This module intentionally stops at the Z=0 ground plane. The implemented
`soccerfield_vitruvian_dlt3d.py` module adds measured goalpost controls and/or
weak player-bbox height verticals to estimate a time-varying DLT3D camera.

---

## When to use which calibration

| Tool | Camera | Per frame? | Use |
|---|---|---|---|
| `soccerfield_calib.py` | fixed / static | one selected frame | Named manual/corrected points on Z=0 |
| `soccerfield_vitruvian_dlt3d.py` | moving broadcast | one row for every supported frame | Pitch plane + goal/player verticals; no raw-coefficient interpolation |
| `fifa_to_dlt.py` (a.k.a. **`fifa dlt-export`**) | **moving broadcast** | **Yes** (one row / frame) | Pan/tilt/zoom — required for real broadcast |
| `rec2d_one_dlt2d.py` | fixed | one row of 8 coeffs | Tripod 2D reconstruction |
| `rec3d_one_dlt3d.py` | fixed (multi-cam) | one row of 11 coeffs/cam | Static lab |
| `rec2d.py` / `rec3d.py` | moving | per-frame DLT | Broadcast |

> For moving cameras, use supplied per-frame `cameras/*.npz` when available.
> Otherwise, the Vitruvian DLT3D module can estimate supported frames directly
> from pitch landmarks and vertical controls. `soccerfield_calib.py` itself
> remains the plane-only component.

---

## Step-by-step (GUI from vailá button)

The SAM workflow can launch a short **Soccer-Field Calib** dialog:

1. Choose a per-video SAM output directory.
2. Optionally choose a named-point pixel CSV; otherwise open `getpixelvideo`.
3. Mark at least six semantic field points on the selected frame.
4. Run the single-frame Z=0 calibration.

For manual clicks only (no CSV yet), use the CLI with `-v` video to open getpixelvideo, or prepare pixels first.

### Legacy: getpixelvideo-only workflow

### Step 1 — Open

Click **Soccer Tools → Soccer-Field Calib** in vailá Frame B.

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

*(Note: the button launcher uses the new CSV-first dialog above; use `uv run python -m vaila.soccerfield_calib -v VIDEO.mp4` for the legacy getpixelvideo-first CLI path.)*

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

# 3a. Static frame: rename/match the visible pN columns to semantic field names,
#     then run soccerfield_calib.py with --frame 0.
# 3b. Moving camera: combine the 32-point CSV with SAM bbox verticals:
uv run python -m vaila.soccerfield_vitruvian_dlt3d \
  --field-pixels out_kps/processed_field_kps_*/field_keypoints_getpixelvideo.csv \
  --bbox-bottom out_sam/VIDEO/sam_vaila_bottom.csv \
  --bbox-top out_sam/VIDEO/sam_vaila_top.csv \
  --heights anthropometry_match.csv --output out_calib/

# 4. Reconstruct player pixels → field metres
uv run vaila/rec2d.py \
  --dlt-file out_calib/video.dlt2d \
  --input-dir player_pixels/ \
  --output-dir player_world/ --rate 30
```

For **broadcast (moving camera)**, supplied `cameras/*.npz` remain the strongest
route. The Vitruvian route above is the implemented calibration fallback when
only image field evidence and vertical controls are available.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| "Need at least 6 points" | Not enough clicks | Click 8–10 well-distributed kps |
| Huge world error (>5 m) | Wrong correspondence | Verify name ↔ world xy in `--list-keypoints` |
| Fit fine, players still wrong | Camera moves between frames | Use `fifa_to_dlt.py` or `soccerfield_vitruvian_dlt3d.py` per frame |
| `pitch_keypoints.png` index ≠ FIFA name | Two indexing systems | The 32-id system (AI seed) is generic; the 29 FIFA names are the calibration targets — match by visual location |

---

## Related

- `soccerfield_keypoints_ai.py` — AI seed for the 32 keypoints
- `getpixelvideo.py` — manual click / refine
- `fifa_to_dlt.py` — per-frame DLT for moving camera
- `soccerfield_vitruvian_dlt3d.py` — time-varying pitch + Vitruvian DLT3D
- `vaila/rec2d.py`, `vaila/rec3d.py` — pixel → world reconstruction
- `vaila/dlt2d.py`, `vaila/dlt3d.py` — DLT math
- `vaila/help/vaila_sam.html` — FIFA pipeline + broadcast section

Updated: 11 August 2026.
