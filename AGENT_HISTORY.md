# Agent Continuation Log — vailá ↔ FIFA Skeletal Tracking Light 2026

Last update: 2026-04-25
Working clone: `/home/preto/data/vaila`
Companion repo: `/home/preto/data/FIFA/FIFA-Skeletal-Tracking-Starter-Kit-2026`

---

## Goal

Use **vailá** modules (`vaila_sam.py`, `soccerfield_keypoints_ai.py`,
`soccerfield_calib.py`, plus the `fifa_*` pipeline) to improve the
**FIFA Skeletal Tracking Light 2026** baseline that runs from the
official starter kit at `/home/preto/data/FIFA`.

Strategy:

1. Run **SAM 3** to get clean per-player video masks (`vaila_sam.py`).
2. Detect **soccer pitch keypoints** per frame to refine camera
   calibration on broadcast footage (`soccerfield_keypoints_ai.py`).
3. Convert FIFA `cameras/*.npz` → per-frame DLT2D/DLT3D and feed the
   reconstruction pipeline (`vaila/fifa_to_dlt.py`,
   `rec2d.py` / `rec3d.py`).
4. Optionally run the SAM 3D Body branch (`fifa baseline`) and
   pack the official submission.

---

## Current state of `/home/preto/data/vaila`

- Active template: **`pyproject_linux_cuda12.toml`** (CUDA 12.8, PyTorch 2.9.1+cu128).
- `.venv` at `/home/preto/data/vaila/.venv` is healthy:
  - `torch.cuda.is_available() == True` (RTX class GPU detected)
  - `sam3` package importable
  - `ultralytics`, `inference`, `supervision` installed
- Extras synced: `--extra gpu --extra sam`.
- SAM 3 weights present:
  - `vaila/models/sam3/sam3.pt`
  - `vaila/models/sam3/sam3.1_multiplex.pt`
- Fix applied during the session: `opencv-python==4.10.0.84`
  reinstalled (previous cv2 install was missing `cv2.VideoCapture`).
- Smoke test SAM 3 succeeded on 1 video, 32 frames:
  - input  : `data/videos/ARG_CRO_000737.mp4` (FIFA starter kit)
  - output : `/home/preto/data/FIFA/outputs_sam3_smoke/processed_sam_20260425_183925/`

### Pending in the FIFA pipeline

- `--extra fifa` **not yet synced** (needed for `fifa baseline`,
  `fifa preprocess`, SAM 3D Body). Run when ready:

  ```bash
  bash bin/setup_fifa_sam3d.sh   # clones sam_3d_body + downloads gated weights
  uv sync --extra gpu --extra sam --extra fifa
  ```

- A local YOLO-pose model **does** exist at
  `vaila/models/soccerfield_keypoints_yolo/football_pitch32_best.pt`
  (`.pt` 6.8 MB, `.onnx` 13 MB), task=pose, kpt=[32,3].
  **It is collapsed** (sanity test 2026-04-25 21:30): boxes are
  fine but every one of the 32 keypoints sits in a ~30 px cluster
  near the box centre. Symptom matches the
  `soccer-field-keypoints-yolo` SKILL collapse warning
  (mosaic=1 + small imgsz).
- Vendored training dataset is on disk and ready for retraining:
  `vaila/models/hf_datasets/football-pitch-detection/data/data.yaml`
  (kpt_shape `[32, 3]`, `flip_idx` defined; 255 train + 34 val).
- Hugging Face CLI (`hf`) is broken in this venv (missing `typer`
  dep when invoked through `uv run`). `vaila_sam.py --download-weights`
  works around it for SAM 3 video; for SAM 3D Body / FIFA you still
  need a working HF auth — easiest is to log in once with the user's
  global Python env or set `HF_TOKEN` in `.env`.

---

## What was completed in this session

1. Verified `/home/preto/data/vaila` already ships:
   - `vaila/vaila_sam.py` (SAM 3 video segmentation + `fifa` subcommands)
   - `vaila/soccerfield_keypoints_ai.py` (Ultralytics + Roboflow backend, video CSV export)
   - `vaila/soccerfield_calib.py` (manual / DLT2D pitch calibration)
   - `vaila/fifa_skeletal_pipeline.py`, `vaila/fifa_to_dlt.py`,
     `vaila/fifa_bootstrap.py`, `vaila/fifa_starter_lib/`
2. Switched `pyproject.toml` to the Linux CUDA template.
3. Synced GPU + SAM extras with `uv sync --extra gpu --extra sam`.
4. Downloaded SAM 3 video weights via `vaila/vaila_sam.py --download-weights`.
5. Fixed a broken OpenCV install with
   `uv pip install --reinstall opencv-python==4.10.0.84`.
6. Ran a successful SAM 3 smoke test on one FIFA video.

## Known issues / decisions

- `hf auth login` fails inside the venv (`No module named 'typer'`).
  `uv add typer` did not actually expose `typer` to `uv run`. For
  SAM 3 video weights this is irrelevant (the loader handles it), but
  for **SAM 3D Body** you must authenticate before
  `bash bin/setup_fifa_sam3d.sh`. Recommended workaround:

  ```bash
  pipx install huggingface_hub        # or use system Python
  hf auth login                       # paste HF token (read access ok)
  # then re-run uv sync if needed
  ```

- `soccerfield_calib.py` expects semantic keypoint names
  (`center_field`, `left_penalty_spot`, …). Detector outputs
  generic `p1, p2, …`. For now, manual calibration is the safer path
  until you publish a mapping or train a YOLO-pose model with the
  vendored 32-keypoint schema.

---

## Next concrete actions

1. **Wait for the 400-epoch retrain** (PID 2070719, ETA ~50 min).
   Monitor with:

   ```bash
   tail -1 vaila/models/runs/pose_fifa/pitch32_recipeA_400ep/results.csv \
     | awk -F',' '{printf "epoch=%s box_mAP50=%s pose_mAP50=%s\n",$1,$11,$15}'
   ```

   Final `best.pt` will land at
   `vaila/models/runs/pose_fifa/pitch32_recipeA_400ep/weights/best.pt`.

2. Once training is done (GPU free), run **SAM 3 in batch** over all
   20 FIFA videos:

   ```bash
   uv run vaila/vaila_sam.py \
     -i /home/preto/data/FIFA/FIFA-Skeletal-Tracking-Starter-Kit-2026/data/videos \
     -o /home/preto/data/FIFA/outputs_sam3 \
     -t person --max-frames 64 --max-input-long-edge 1280 \
     --postprocess-points foot
   ```

3. Re-run keypoints on every video with the **new `best.pt`**:

   ```bash
   for f in /home/preto/data/FIFA/FIFA-Skeletal-Tracking-Starter-Kit-2026/data/videos/*.mp4; do
     uv run python -m vaila.soccerfield_keypoints_ai \
       --mode video -i "$f" -o /home/preto/data/FIFA/outputs_pitch_kps_final \
       --backend ultralytics \
       --weights vaila/models/runs/pose_fifa/pitch32_recipeA_400ep/weights/best.pt \
       --imgsz 1280 --conf 0.30 --draw-min-conf 0.40 \
       --device 0 --stride 1 --overlay-video
   done
   ```

4. Map detector indices `kp_00..kp_31` → semantic point names from
   `vaila/models/soccerfield_ref3d.csv` (use
   `vaila/models/hf_datasets/football-pitch-detection/pitch_keypoints.png`
   as reference layout).
5. Feed mapped points to `soccerfield_calib.py` to fit per-frame
   DLT2D, then `vaila/vaila_sam.py fifa dlt-export` for `rec2d.py /
   rec3d.py`.
6. Only then attack `fifa baseline` (needs
   `bash bin/setup_fifa_sam3d.sh && uv sync --extra fifa`).

See `HISTORY.md` for the chronological log and
`.claude/skills/fifa-vaila-continuation/SKILL.md` for ready-to-run
command blocks.
