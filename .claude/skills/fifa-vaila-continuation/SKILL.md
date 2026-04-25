---
name: fifa-vaila-continuation
description: Resume work on the FIFA Skeletal Tracking Light 2026 challenge using vailá modules (SAM 3, soccer pitch keypoints, calibration, per-frame DLT). Use whenever the user reopens the project after closing the terminal, asks "where did we stop on FIFA?", or wants the next concrete commands to run from `/home/preto/data/vaila` with `.venv` already created.
---

# FIFA + vailá Continuation Skill

Companion files:

- `/home/preto/data/vaila/AGENT_HISTORY.md` — what was done, current state, decisions.
- `/home/preto/data/vaila/HISTORY.md` — chronological log of the session.
- `.claude/skills/fifa-skeletal-tracking/SKILL.md` — full FIFA pipeline reference.
- `.claude/skills/sam3-video/SKILL.md` — SAM 3 video segmentation reference.
- `.claude/skills/soccer-field-keypoints-yolo/SKILL.md` — pitch-keypoint training/inference.
- `.claude/skills/sports-field-visualization/SKILL.md` — field rendering helpers.

Companion repo (the official starter kit):
`/home/preto/data/FIFA/FIFA-Skeletal-Tracking-Starter-Kit-2026`.

---

## Objective

Continue from a known-good baseline and execute the improvement path:

1. SAM 3 per-player video masks
2. Per-frame **soccer pitch keypoints** for camera refinement
3. Per-frame **DLT** export for `rec2d.py` / `rec3d.py`
4. (Optional) `fifa baseline` (SAM 3D Body) → `fifa pack` for Codabench

---

## Known-good baseline (validated 2026-04-25)

- `pyproject.toml` ← `pyproject_linux_cuda12.toml` (PyTorch 2.9.1+cu128).
- `uv sync --extra gpu --extra sam` succeeded.
- `vaila/models/sam3/sam3.pt` and `sam3.1_multiplex.pt` present.
- `cv2.VideoCapture` is back after `uv pip install --reinstall opencv-python==4.10.0.84`.
- SAM 3 smoke test: 32 frames of `ARG_CRO_000737.mp4` written to
  `/home/preto/data/FIFA/outputs_sam3_smoke/processed_sam_20260425_183925/`.

---

## Resume Sequence (paste in this order)

### 0. Enter the project

```bash
cd /home/preto/data/vaila
source .venv/bin/activate     # optional — `uv run` works without it
```

### 1. Re-confirm CUDA template + extras (idempotent)

```bash
bash bin/use_pyproject_linux_cuda.sh
uv sync --extra gpu --extra sam
```

### 2. Make sure SAM 3 weights are still there

```bash
uv run vaila/vaila_sam.py --download-weights   # no-op if already present
ls vaila/models/sam3/sam3.pt
```

### 3. Quick smoke test (1 video, 32 frames) — only if a fresh sanity check is needed

```bash
uv run vaila/vaila_sam.py \
  -i /home/preto/data/FIFA/FIFA-Skeletal-Tracking-Starter-Kit-2026/data/videos/ARG_CRO_000737.mp4 \
  -o /home/preto/data/FIFA/outputs_sam3_smoke \
  -t person --max-frames 32
```

### 4. Run SAM 3 batch over all FIFA videos

Default uses subprocess-per-video isolation (clean GPU between clips):

```bash
uv run vaila/vaila_sam.py \
  -i /home/preto/data/FIFA/FIFA-Skeletal-Tracking-Starter-Kit-2026/data/videos \
  -o /home/preto/data/FIFA/outputs_sam3 \
  -t person \
  --max-frames 64 \
  --max-input-long-edge 1280 \
  --postprocess-points foot
```

Useful flags:

- `--frame-by-frame` — lower VRAM (CUDA only).
- `--no-isolate-batch` — same-process batch (debug only).
- `-t "team in red"`, `-t goalkeeper`, etc. — text presets accepted by SAM 3.

### 5. Soccer-pitch keypoints (need EITHER Roboflow OR a YOLO `best.pt`)

#### 5A. Roboflow backend (no local weights needed)

```bash
export ROBOFLOW_API_KEY="YOUR_KEY"
uv run python -m vaila.soccerfield_keypoints_ai \
  --mode video \
  -i /home/preto/data/FIFA/FIFA-Skeletal-Tracking-Starter-Kit-2026/data/videos/ARG_CRO_000737.mp4 \
  -o /home/preto/data/FIFA/outputs_pitch_kps \
  --backend roboflow \
  --roboflow-model-id football-field-detection-f07vi/14 \
  --conf 0.3 --draw-min-conf 0.05 \
  --stride 1 --max-frames 300 \
  --overlay-video
```

#### 5B. Local Ultralytics YOLO-pose

```bash
uv run python -m vaila.soccerfield_keypoints_ai \
  --mode video \
  -i /home/preto/data/FIFA/FIFA-Skeletal-Tracking-Starter-Kit-2026/data/videos/ARG_CRO_000737.mp4 \
  -o /home/preto/data/FIFA/outputs_pitch_kps \
  --backend ultralytics \
  --weights /ABS/PATH/TO/best.pt \
  --imgsz 1280 --conf 0.3 --draw-min-conf 0.05 \
  --device 0 --stride 1 --overlay-video
```

To train a `best.pt` from the vendored dataset, follow
`.claude/skills/soccer-field-keypoints-yolo/SKILL.md`
(`vaila/models/hf_datasets/football-pitch-detection/data/data.yaml`,
`kpt_shape: [32, 3]`, `imgsz=1280`, `mosaic=0`, `erasing=0`).

### 6. Calibration (manual first, automatic later)

```bash
# List the semantic point names accepted by the calibrator
uv run python -m vaila.soccerfield_calib --list-keypoints

# Manual interactive calibration on a single FIFA clip
uv run python -m vaila.soccerfield_calib \
  -v /home/preto/data/FIFA/FIFA-Skeletal-Tracking-Starter-Kit-2026/data/videos/ARG_CRO_000737.mp4 \
  -o /home/preto/data/FIFA/outputs_pitch_calib \
  --data-root /home/preto/data/FIFA/FIFA-Skeletal-Tracking-Starter-Kit-2026/data
```

> The detector outputs generic `p1, p2, …`. Until a mapping
> (`p_i` → semantic name) is published, the safe path is manual
> calibration on a few key frames and reuse of the homography for the
> whole clip when the camera is roughly static.

### 7. Per-frame DLT for `rec2d.py` / `rec3d.py`

Once the FIFA `cameras/*.npz` is populated (either from the
official dataset or from `fifa baseline --export-camera`):

```bash
uv run vaila/vaila_sam.py fifa dlt-export \
  --cameras-dir /home/preto/data/FIFA/FIFA-Skeletal-Tracking-Starter-Kit-2026/data/cameras \
  --output-dir   /home/preto/data/FIFA/outputs_dlt_per_frame
```

Use the resulting per-frame `.dlt2d` / `.dlt3d` with
`vaila/rec2d.py` and `vaila/rec3d.py` (broadcast / moving cameras).
For genuinely fixed cameras, use `rec2d_one_dlt2d.py` /
`rec3d_one_dlt3d.py` instead.

### 8. (Optional) Full FIFA pipeline + Codabench submission

```bash
# Once and only once (clones sam_3d_body + downloads gated weights)
uv run hf auth login                        # see workaround below if hf is broken
bash bin/setup_fifa_sam3d.sh
uv sync --extra gpu --extra sam --extra fifa

uv run vaila/vaila_sam.py fifa bootstrap \
  --videos-dir /home/preto/data/FIFA/FIFA-Skeletal-Tracking-Starter-Kit-2026/data/videos \
  --data-root  /home/preto/data/FIFA/FIFA-Skeletal-Tracking-Starter-Kit-2026/data
uv run vaila/vaila_sam.py fifa prepare    --data-root /home/preto/data/FIFA/FIFA-Skeletal-Tracking-Starter-Kit-2026/data
uv run vaila/vaila_sam.py fifa boxes      --data-root /home/preto/data/FIFA/FIFA-Skeletal-Tracking-Starter-Kit-2026/data
uv run vaila/vaila_sam.py fifa preprocess --data-root /home/preto/data/FIFA/FIFA-Skeletal-Tracking-Starter-Kit-2026/data
uv run vaila/vaila_sam.py fifa baseline   --data-root /home/preto/data/FIFA/FIFA-Skeletal-Tracking-Starter-Kit-2026/data \
                                          --output    /home/preto/data/FIFA/outputs/submission_full.npz \
                                          --export-camera
uv run vaila/vaila_sam.py fifa dlt-export --cameras-dir /home/preto/data/FIFA/FIFA-Skeletal-Tracking-Starter-Kit-2026/data/cameras \
                                          --output-dir  /home/preto/data/FIFA/outputs/dlt_per_frame
uv run vaila/vaila_sam.py fifa pack       --submission-full /home/preto/data/FIFA/outputs/submission_full.npz \
                                          --data-root       /home/preto/data/FIFA/FIFA-Skeletal-Tracking-Starter-Kit-2026/data \
                                          --output-dir      /home/preto/data/FIFA/outputs \
                                          --split val
```

---

## Recovery / known issues

- **`hf auth login` fails inside `.venv`** with
  `ModuleNotFoundError: No module named 'typer'`.
  Either set `HF_TOKEN=...` in `.env`, or run `hf auth login` from a
  separate `pipx install huggingface_hub` install.
- **`cv2.VideoCapture` missing** after a fresh sync:
  `uv pip install --reinstall opencv-python==4.10.0.84`.
- **SAM 3 batch CUDA OOM cascade**: already mitigated in `vaila_sam.py`
  via subprocess-per-video isolation. If you still OOM, add
  `--frame-by-frame` and lower `--max-input-long-edge`.

---

## Next-step priorities (highest expected MPJPE win first)

1. **Per-frame camera refinement** with pitch keypoints (Section 5+6).
2. **Cleaner player boxes/masks** with SAM 3 batch (Section 4).
3. Only then touch `fifa baseline` and 3D refinement (Section 8).
