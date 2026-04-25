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
