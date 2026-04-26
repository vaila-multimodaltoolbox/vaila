# Session History — vailá ↔ FIFA Skeletal Tracking Light 2026

Date: 2026-04-25
Repo: `/home/preto/data/vaila` (branch `main`)

This file is the chronological log of what happened in this terminal
session, so any future agent (Cursor / Claude Code / Antigravity /
Windsurf / warp.dev) can resume without reading the whole transcript.

---

## 1. Context before the session

- FIFA challenge starter kit already cloned and partially tuned at
  `/home/preto/data/FIFA/FIFA-Skeletal-Tracking-Starter-Kit-2026`:
  - `uv` venv created in that repo
  - dataset wired into `data/` via symlinks
  - `pandas` added to its `pyproject.toml`
  - `main.py` patched to survive videos shorter than annotation arrays
  - baseline (`main.py`, `scripts/run_jobs.sh`) ran end-to-end
- vailá clone at `/home/preto/data/vaila` already shipped:
  - `vaila/vaila_sam.py` (SAM 3 video segmentation + `fifa` subcommands)
  - `vaila/soccerfield_keypoints_ai.py` (YOLO-pose / Roboflow detector)
  - `vaila/soccerfield_calib.py` (manual / DLT2D pitch calibration)
  - `vaila/fifa_skeletal_pipeline.py`, `vaila/fifa_to_dlt.py`,
    `vaila/fifa_bootstrap.py`
  - `pyproject.toml` matching the **CPU** template
  - existing modifications: `pyproject.toml`, `uv.lock`,
    `vaila/getpixelvideo.py`

## 2. Decisions taken

- Use vailá to attack 3 things, in order:
  1. **Per-player masks** with SAM 3 (`vaila_sam.py`)
  2. **Per-frame pitch keypoints** to refine camera calibration on
     broadcast video (`soccerfield_keypoints_ai.py`)
  3. **Per-frame DLT** export to drive `rec2d.py` / `rec3d.py`
     (`vaila_sam.py fifa dlt-export` / `fifa_to_dlt.py`)
- Don't touch SAM 3D Body / `fifa baseline` yet — needs `--extra fifa`
  + gated HF weights, deferred until pitch keypoints are working.

## 3. Commands actually executed in `/home/preto/data/vaila`

```bash
bash bin/use_pyproject_linux_cuda.sh
uv sync --extra gpu --extra sam
uv run vaila/vaila_sam.py --download-weights

# OpenCV fix (cv2.VideoCapture missing)
uv pip install --reinstall opencv-python==4.10.0.84

# SAM 3 smoke test (1 video, 32 frames)
uv run vaila/vaila_sam.py \
  -i /home/preto/data/FIFA/FIFA-Skeletal-Tracking-Starter-Kit-2026/data/videos/ARG_CRO_000737.mp4 \
  -o /home/preto/data/FIFA/outputs_sam3_smoke \
  -t person --max-frames 32
```

## 4. Verified results

- `pyproject.toml` now matches `pyproject_linux_cuda12.toml`.
- `.venv` Python: `torch 2.9.1+cu128`, `torch.cuda.is_available() == True`.
- `sam3` package importable.
- Files present:
  - `vaila/models/sam3/sam3.pt`
  - `vaila/models/sam3/sam3.1_multiplex.pt`
- Smoke test output:
  - `/home/preto/data/FIFA/outputs_sam3_smoke/processed_sam_20260425_183854/`
  - `/home/preto/data/FIFA/outputs_sam3_smoke/processed_sam_20260425_183925/`

## 5. Known issues + workarounds

| Issue | Symptom | Workaround |
|-------|---------|-----------|
| `hf` CLI broken inside `.venv` | `ModuleNotFoundError: typer` | Use system / pipx `huggingface_hub`, or set `HF_TOKEN` in `.env`. SAM 3 video does **not** need it (the loader handles weights). |
| Old OpenCV without `VideoCapture` | `vaila_sam.py` failed at first run | `uv pip install --reinstall opencv-python==4.10.0.84` |
| No local YOLO `best.pt` for pitch keypoints | `soccerfield_keypoints_ai --backend ultralytics --weights …` impossible | Use `--backend roboflow` with API key, OR train one (see `.claude/skills/soccer-field-keypoints-yolo/SKILL.md`). |
| Detector keypoint names are `p1, p2, …` | `soccerfield_calib.py` expects semantic names | Calibrate manually first OR write a mapping CSV before automatic export. |

## 6. Pending / not yet executed

- `bash bin/setup_fifa_sam3d.sh` (clones `sam_3d_body/` + downloads
  gated `facebook/sam-3d-body-dinov3` weights into
  `vaila/models/sam-3d-dinov3/`).
- `uv sync --extra gpu --extra sam --extra fifa` (FIFA Lightning stack).
- SAM 3 batch over all videos in
  `/home/preto/data/FIFA/.../data/videos/`.
- Soccer-pitch keypoint detection on FIFA broadcast clips.
- `fifa baseline` / `fifa dlt-export` / `fifa pack` for the
  Codabench submission.

## 7. File-system state at end of session

Modified (vs `main`):

```
 M pyproject.toml         # switched to CUDA template
 M uv.lock                # regenerated for CUDA + sam extra
 M vaila/getpixelvideo.py # was already modified before the session
?? AGENT_HISTORY.md
?? HISTORY.md
?? .claude/skills/fifa-vaila-continuation/SKILL.md
```

Nothing was committed during this session.

---

## 8. Update — 2026-04-25 21:30 (sanity tests)

### Pitch keypoints — `football_pitch32_best.pt` is COLLAPSED

- Found the local trained model at
  `vaila/models/soccerfield_keypoints_yolo/football_pitch32_best.pt`
  (also `.onnx`, ~13 MB). Task=pose, names={0:'football_pitch'},
  kpt_shape=[32,3].
- Smoke test (5 frames, stride 30) on `ARG_CRO_000737.mp4`:
  output dir `/home/preto/data/FIFA/outputs_pitch_kps/processed_field_kps_20260425_213008/`.
- Direct YOLO predict on frame 0 (`imgsz=1280`, `conf=0.05`, `device=0`):
  - boxes OK: 3 detections covering the field with conf 0.89 / 0.66 / 0.29.
  - **all 32 keypoints collapsed inside a ~30 px cluster** of the
    first box (box 0): `x ∈ [128, 160]`, `y ∈ [787, 822]` —
    none of the 32 keypoints is ever above `y=787`, even though the
    image is 1080 high and the upper field is clearly visible.
  - Same pattern across frames 0/30/60/120: the cluster moves with
    the box centre but the geometry is gone.
- Diagnosis: classic **mosaic-collapse** described in
  `.claude/skills/soccer-field-keypoints-yolo/SKILL.md`
  (naive training with `mosaic=1.0` + small `imgsz` keeps box mAP
  high while keypoints collapse onto a single point).
- Conclusion: this `best.pt` is **not usable** for FIFA. Need to
  retrain with the recipe from the skill (or fall back to Roboflow
  / manual homography).

### Dataset for retraining (already on disk)

- `vaila/models/hf_datasets/football-pitch-detection/`
  - `data/data.yaml` (`kpt_shape: [32, 3]`, `flip_idx` defined)
  - `data/train/images/` — **255** images
  - `data/valid/images/` — **34** images
  - `data/test/images/` (untested)
  - `pitch_keypoints.png` (reference layout)

### SAM 3 tracking — WORKING

- 60-frame test on `ARG_CRO_000737.mp4`:
  - input downscaled 1920x1080 → 1280x720 (VRAM cap)
  - propagation ~1.84 s/iter on the local CUDA GPU
  - output dir
    `/home/preto/data/FIFA/outputs_sam3_test1/processed_sam_20260425_213120/ARG_CRO_000737/`
- Detected **37 person IDs** across 59 frames; `sam_points.csv`
  written with stable `p1..p37` foot keypoints (mode=foot).
- Files produced per video:
  - `ARG_CRO_000737_sam_overlay.mp4` (5.4 MB)
  - `masks/` (per-frame PNGs)
  - `sam_id_map.csv`, `sam_frames_meta.csv`, `sam_points.csv`
- Verdict: pipeline is solid; safe to scale to all 20 FIFA videos.

### Action plan

1. **Retrain pitch keypoint model** with the proven recipe
   (`yolo26s-pose`, `imgsz=1280`, `mosaic=0`, `mixup=0`,
   `close_mosaic=0`, `erasing=0`, `pose=25`, `kobj=2`, `epochs=800`,
   `batch=8`).
2. While training runs, scale **SAM 3** to the full 20-video batch
   (`-i .../data/videos`, `--max-frames 64`, isolation ON by default).
3. Once a non-collapsed `best.pt` exists, re-run pitch detection,
   then map detector indices → semantic names and feed
   `soccerfield_calib.py` / `fifa_to_dlt.py`.

---

## 9. Update — 2026-04-25 21:42 (retrain proof-of-concept WORKED)

### 50-epoch retrain (recipe A, proof-of-concept)

Command:

```bash
uv run yolo pose train \
  model=/home/preto/data/vaila/yolo26s-pose.pt \
  data=/home/preto/data/vaila/vaila/models/hf_datasets/football-pitch-detection/data/data.yaml \
  epochs=50 imgsz=1280 batch=8 \
  mosaic=0.0 mixup=0.0 close_mosaic=0 erasing=0.0 \
  pose=25.0 kobj=2.0 device=0 \
  project=/home/preto/data/vaila/vaila/models/runs/pose_fifa \
  name=pitch32_recipeA_50ep
```

- ~7 s/epoch on RTX 4090, ~12 GB VRAM, batch=8, imgsz=1280.
- Final epoch 50 metrics:
  - **Box mAP50 = 0.830, mAP50-95 = 0.561**
  - **Pose mAP50 = 0.170, mAP50-95 = 0.028**
- Output: `vaila/models/runs/pose_fifa/pitch32_recipeA_50ep/weights/best.pt`
  (and `last.pt`).

### Sanity test on FIFA frame 0 — COLLAPSE FIXED

Direct YOLO predict, `imgsz=1280`, `conf=0.05`:

| Metric | Old `football_pitch32_best.pt` | New `pitch32_recipeA_50ep/best.pt` |
|---|---|---|
| `kp x` range | 128 – 160 (≈30 px) | **0 – 1920 (full width)** |
| `kp y` range | 787 – 822 (≈35 px) | **501 – 1080 (lower half + line)** |
| Distinct 50 px bins / frame | **4** | **26** |
| `kp_max conf` | ~0.54 (uniform) | 0.94 (and 0.001–0.94 spread, expected for under-trained model) |

Conclusion: the recipe escapes the collapse even at 50 epochs and the
detector now produces **field geometry** (with low confidence on most
keypoints — that improves with more training).

CSV demo (CPU inference, since GPU was busy with the 400-ep retrain):
`/home/preto/data/FIFA/outputs_pitch_kps_v2/processed_field_kps_20260425_214224/`.

### 400-epoch retrain — DONE 2026-04-26 08:32 (excellent)

Final epoch 400 metrics:

| Metric | Value |
|---|---|
| **Box mAP50** | **0.960** |
| **Box mAP50-95** | **0.788** |
| **Pose mAP50** | **0.945** |
| **Pose mAP50-95** | **0.813** |

Validated visually on FIFA `ARG_CRO_000737.mp4` frame 0:

- Single high-confidence detection (box conf 0.887)
- 11/32 keypoints with conf ≥ 0.5 (the rest correctly inactive
  because they fall outside the broadcast frame)
- Top kp_13/14/15 with conf 0.996 — correctly aligned with the
  visible side line
- Visual overlay at `/tmp/overlay_frame0_400ep.png` confirms
  keypoints land on the actual field markings (mid-line, centre
  circle, right penalty area, lower side line)

Full output (60 frames, stride=5): `/home/preto/data/FIFA/outputs_pitch_kps_final/processed_field_kps_20260426_083336/`

Best.pt path:
`vaila/models/runs/pose_fifa/pitch32_recipeA_400ep/weights/best.pt`

### 400-epoch continuation training (was running, PID 2070719 — finished)

```bash
uv run yolo pose train \
  model=…/pitch32_recipeA_50ep/weights/best.pt \
  data=…/data.yaml \
  epochs=400 imgsz=1280 batch=8 \
  mosaic=0.0 mixup=0.0 close_mosaic=0 erasing=0.0 \
  pose=25.0 kobj=2.0 device=0 patience=80 \
  project=…/runs/pose_fifa name=pitch32_recipeA_400ep
```

- Log: `vaila/models/runs/pose_fifa/train_recipeA_400ep.log`
- Output: `vaila/models/runs/pose_fifa/pitch32_recipeA_400ep/`
- ETA ~50 min on RTX 4090.
- **Note:** Ultralytics treats the warm-start as a fresh schedule
  (epoch 1/400), so `pose_mAP50` momentarily drops back to 0
  while learning rate is high; expect recovery after ~30–80 epochs.
  Monitor `results.csv`:

  ```bash
  tail -1 vaila/models/runs/pose_fifa/pitch32_recipeA_400ep/results.csv \
    | awk -F',' '{printf "epoch=%s box_mAP50=%s pose_mAP50=%s\n",$1,$11,$15}'
  ```

### What to do while the 400-ep retrain finishes

- **Cannot** run SAM 3 in parallel: training uses ~12 GB VRAM, SAM 3
  peaks at ~13 GB. Wait or run SAM 3 with `--frame-by-frame`.
- **Can** run keypoint inference on **CPU** (very small model,
  ~5 s for 5 frames @ imgsz=1280): use `--device cpu`.

### Outputs available right now

| What | Path |
|---|---|
| Old (collapsed) keypoints CSV | `/home/preto/data/FIFA/outputs_pitch_kps/processed_field_kps_20260425_213008/` |
| New (50-ep) keypoints CSV | `/home/preto/data/FIFA/outputs_pitch_kps_v2/processed_field_kps_20260425_214224/` |
| New best.pt (50 ep, geometry recovered) | `vaila/models/runs/pose_fifa/pitch32_recipeA_50ep/weights/best.pt` |
| SAM 3 tracking (60 frames, 37 person IDs) | `/home/preto/data/FIFA/outputs_sam3_test1/processed_sam_20260425_213120/ARG_CRO_000737/` |

### Backup / fallback solutions if 400-ep retrain plateaus

1. **Increase pose loss weight further** (e.g. `pose=40`, `kobj=4`)
   and reset from `best.pt` 50ep with `epochs=300`.
2. **Switch to `yolo26m-pose.pt` or `yolo26x-pose.pt`** with smaller
   `lr0=1e-4`, `amp=False` to avoid NaN.
3. **Add data augmentation that preserves keypoints** (`degrees=10`,
   `translate=0.1`, `scale=0.5`) but **never** `mosaic` or `mixup`.
4. **Roboflow backend** as drop-in (no local training):
   `--backend roboflow --roboflow-model-id football-field-detection-f07vi/14`
   with `ROBOFLOW_API_KEY` in env.
5. **Manual homography**: pick 4 frames per video, click 6+ pitch
   landmarks each, fit `cv2.findHomography`, interpolate the rest.
   Use `python -m vaila.soccerfield_calib --list-keypoints`.
6. **External SOTA**: integrate
   [PnLCalib](https://github.com/MM4SPA/PnLCalib) or
   [No-Bells-PnL](https://github.com/mguti97/No-Bells-Just-Whistles)
   as alternative pitch detectors (both Apache-2 / MIT).

