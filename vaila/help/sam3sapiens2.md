# SAM3+Sapiens2 (`sam3sapiens2.py`)

## Module information

- **Category:** Markerless 2D / Meta (Facebook)
- **Version:** 0.3.120
- **Updated:** 2026-09-03
- **GUI:** Frame B → **Markerless 2D** → **SAM3+Sapiens2**
- **CLI:** Yes
- **Retomada:** `--resume /caminho/processed_sam3sapiens2_...` reaproveita somente vídeos e resultados SAM com cobertura completa comprovada; informe também `-i` com a pasta original. Sem `--resume`, uma execução repetida com o mesmo `-i`/`-o` já retoma sozinha o `processed_sam3sapiens2_*` correspondente (auto-resume); use `--fresh` para forçar uma pasta nova.
- **Selected-ID rerender:** after a run, use [sam3sapiens2_visualize](sam3sapiens2_visualize.md) (GUI **SAM3+Sapiens2 Visualize ID**, or CLI `--id` / interactive prompt). Overlay matches SAM3 contour + Sapiens2 left/right skeleton colors.

## What this pipeline changes

This is a real mixed pipeline, not just two independent runs in sequence:

1. **SAM3 runs first** and defines each person's bounding box, silhouette, score, and persistent `obj_id`.
2. **Sapiens2 runs only inside those SAM-guided boxes.** Its DETR person detector is disabled and never loaded.
3. The Sapiens2 input for each person keeps the original foreground inside the SAM contour and blurs competing background inside the crop.
4. Keypoints outside the dilated SAM silhouette remain available but their confidence is attenuated (default factor `0.25`).
5. Identity is inherited without a second assignment: `stable_id == person_id == sam_obj_id`.

This avoids duplicate person detection, reduces false positives from the full frame, and prevents Sapiens2 Re-ID from swapping identities already established by SAM3.

## Requirements

Both optional CUDA pipelines must already work:

```bash
bash bin/setup_pyproject.sh --target=linux-cuda --extras=gpu,sam,sapiens --yes
bash bin/setup_sapiens2.sh
uv run hf auth login
uv run vaila/vaila_sam.py --download-weights
```

Inference requires an NVIDIA CUDA GPU. SAM3/Sapiens2 model licenses remain Meta licenses, separate from vailá's AGPL source license.

`bin/setup_sapiens2.sh` only fetches the `1b` pose checkpoint. When you pick another size (`0.4b`, `0.8b`, `5b`), the run downloads the missing checkpoint automatically before the SAM3 stage starts, so a batch never burns GPU time only to fail at the pose stage. To fetch one ahead of time:

```bash
uv run vaila/vaila_sapiens.py --download-weights --model 0.4b
```

## GUI

Open **Markerless 2D → SAM3+Sapiens2**. On the input row:

- **Dir…** — pick a folder; every video in that folder is processed in batch
  (non-recursive; `.mp4` / `.avi` / `.mov` / `.mkv` / `.webm`);
- **File…** — pick a single clip.

Also select:

- an output parent;
- optionally an existing `processed_sam_*` directory;
- SAM text prompt, Sapiens2 model, stride, CUDA device, keypoint threshold, pose batch size;
- bbox padding and contour margin.

Click **Run**. The terminal prints how many videos were queued, then the
complete `>> Equivalent CLI` command inside a highlighted bold-yellow banner
(plain text when the terminal isn't interactive, e.g. redirected to a log
file, or `NO_COLOR` is set) — copy/paste it to repeat this exact run headlessly
later without reopening the GUI.
Running `uv run python -u vaila/sam3sapiens2.py` without `-i` and `-o` also
opens this settings window. The window is centered and raised above the
terminal; the terminal prints confirmation as soon as the dialog is mapped.
For a terminal-only run, always provide both `-i/--input` and `-o/--output`
(`-i` may be a folder for the same batch behaviour).

If **Existing SAM results** is empty, every video is processed as:

```text
SAM3 child process → child exits/VRAM is clean → Sapiens2-from-SAM worker
```

The transition is guarded by a GPU recovery barrier: descendants are terminated and free VRAM must return to the pre-SAM baseline before Sapiens2 can load. `sam_frames_meta.csv` must contain every source frame; otherwise Sapiens2 is not loaded.

If it points to an existing batch, SAM3 is skipped and its per-video subdirectory is matched by video stem.

## CLI

### Full SAM3 then guided Sapiens2

```bash
uv run python -u vaila/sam3sapiens2.py \
  -i /path/to/videos \
  -o /path/to/output \
  -t person \
  --model 1b \
  --stride 1 \
  --device 0 \
  --kpt-thr 0.3 \
  --pose-batch-size 2 \
  --bbox-padding 0.12 \
  --contour-margin 8
```

### Resume a partial / failed SAM3+Sapiens2 run

```bash
uv run python -u vaila/sam3sapiens2.py \
  -i /path/to/videos \
  --resume /path/to/processed_sam3sapiens2_YYYYMMDD_HHMMSS \
  --model 1b
```

`--resume` keeps the same timestamped folder:

- videos with a valid, `completed=true` `sam3sapiens2_summary.json` whose frame count matches the source are skipped;
- videos whose `sam3/sam_frames_meta.csv` proves complete frame coverage skip SAM3 and run only Sapiens2;
- expected frame counts come from **decodable** frames (v0.3.120+), not raw container `nb_frames`, so inflated metadata no longer blocks Sapiens2 after a complete chunked SAM merge;
- a mere `sam_tracks.csv` is not enough: partial SAM runs are rejected and rerun;
- failed videos clear stale `_chunks` / `FAILED_*.txt` and re-run SAM3 via the CUDA-clean coordinator, then Sapiens2.

GUI: no field for this (removed in v0.3.107) — re-running with the same
output parent auto-resumes automatically (see below). `--resume` for an
explicit path override is CLI-only.

### Automatic resume (default, no flag needed)

Re-running the same command (same `-i` and `-o`) auto-detects a matching
`processed_sam3sapiens2_*` directory already under `-o` — via a
`BATCH_INPUT.json` marker written at batch start, or the completed run's own
`sam3sapiens2_batch_summary.json` — and resumes it exactly like `--resume`
would, printing:

```text
Auto-resume: found matching run, reusing /path/to/processed_sam3sapiens2_...
Resume: 3/8 videos already completed, 5 remaining
[SKIP] Already processed: clip_003.mp4
```

Pass `--fresh` to ignore any match and start a brand-new timestamped output
directory instead. `--fresh` and `--resume` are mutually exclusive.

### Reuse a completed SAM3 batch

```bash
uv run python -u vaila/sam3sapiens2.py \
  -i /path/to/videos \
  -o /path/to/output \
  --sam-results /path/to/processed_sam_20260730_113024 \
  --model 1b --stride 1
```

`--sam-results` accepts either the batch parent or, for a single input video, the direct directory containing `sam_tracks.csv` and `sam_contours.json`.

### Dry-run (no model inference)

```bash
uv run python -u vaila/sam3sapiens2.py \
  -i /path/to/videos -o /tmp/check \
  --sam-results /path/to/processed_sam_... --dry-run
```

The dry-run validates video-to-SAM matching, bbox columns, contour metadata, and prints the effective plan.

## Precision and performance controls

| Flag | Default | Meaning |
|------|---------|---------|
| `--bbox-padding` | `0.12` | Context added around the tight SAM contour box before Sapiens2. |
| `--contour-margin` | `8` px | Dilation around the silhouette for crop focus and keypoint validation. |
| `--outside-contour-factor` | `0.25` | Confidence multiplier for a keypoint outside the dilated contour. |
| `--min-sam-score` | `0.0` | Optional SAM detection-score filter. |
| `--min-sam-area` | `64` px | Reject tiny mask fragments. |
| `--max-persons` | `32` | Safety cap; highest score×area objects are kept when exceeded. |
| `--no-contour-focus` | off | Keep SAM boxes/IDs but feed the unmodified image crop to Sapiens2. |
| `--stride` | `1` | Pose every frame. Gaps use the nearest pose transformed to the current SAM box. |
| `--flip-test` | off | Higher-quality Sapiens2 pass at extra compute/VRAM. |
| `--no-overlay` | off | Skip the combined MP4. |

SAM-specific pass-through flags include `--sam-frame`, `--sam-checkpoint`, `--sam-max-frames`, `--sam-max-input-long-edge`, and `--keep-sam-masks`.

## Outputs

Each run creates `processed_sam3sapiens2_<timestamp>/<video_stem>/`:

- `<video>_sam3sapiens2_overlay.mp4` — Sapiens2 skeleton plus SAM contour, bbox, confidence, and `SAM #ID`.
- `<video>_sam3sapiens2_predictions.json` — 308 keypoints, SAM provenance, both SAM and pose boxes, contour checks, and `detr_loaded: false`.
- `<video>_sam3sapiens2_vaila.csv` — long frame/person/keypoint table.
- `sam3sapiens2_id_audit.csv` — per-frame evidence that `sam_obj_id == stable_id`.
- `<video>_markers.csv`, `sapiens_vaila_*.csv`, `sapiens_points.csv` — getpixelvideo and REC2D/REC3D formats.
- `sapiens_id_map.csv`, `sapiens_bbox_tracks.csv` — stable slots and SAM-compatible bbox tracks.
- `<video>_id_NN_sapiens_pose.csv` — wide 308-keypoint file for each SAM identity.
- `sam3sapiens2_summary.json` and batch summary JSON.
- `README_sam3sapiens2.txt` with run provenance.
- `FAILED_sam3sapiens2.txt` only on failure.

When SAM3 is run by the combined pipeline, its artifacts are stored under `<video_stem>/sam3/`. When `--sam-results` is used, outputs reference the existing SAM directory without duplicating it.

## Identity guarantee

The combined pipeline does not call DETR, `SapiensTemporalLinker`, bidirectional geometric Re-ID, or appearance Re-ID. The SAM object ID is written directly to all Sapiens `raw_id`, `temporal_id`, `stable_id`, and CSV `person_id` fields. This invariant is also written to `sam3sapiens2_id_audit.csv` for downstream verification.
