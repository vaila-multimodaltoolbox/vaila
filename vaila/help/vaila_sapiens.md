# vaila_sapiens.py

## Module Information

- **Category:** Markerless 2D / Meta (Facebook)
- **File:** `vaila/vaila_sapiens.py`
- **Version:** 0.3.76
- **Updated:** 2026-07-07
- **GUI Interface:** Yes
- **CLI Interface:** Yes

## Description

Top-down **Sapiens2 Pose** (308 keypoints, Sociopticon topology) on video.
Uses Meta's DETR person detector + Sapiens2 ViT pose head. Integrated into
vailá alongside SAM 3 under **Frame B → YOLO + FB → Sapiens2 Pose**.

> **License:** Sapiens2 weights and code use Meta's **Sapiens2 License** (not
> AGPL). Download only from Hugging Face after accepting the model terms.

## Requirements

- **NVIDIA CUDA** (no CPU/macOS path in this integration)
- **Linux NVIDIA install:** when running `install_vaila_linux.sh` with GPU support,
  answer **Y** to *Install optional Sapiens2 Pose*; then confirm
  `bash bin/setup_sapiens2.sh` when prompted.
- Manual bootstrap:

```bash
uv sync --extra sapiens
bash bin/setup_sapiens2.sh
```

This clones `.local/third_party/sapiens2/` (gitignored) and downloads:

- `vaila/models/sapiens2/pose/sapiens2_1b_pose.safetensors`
- `vaila/models/sapiens2/detector/detr-resnet-101-dc5/`

## Default model (RTX 4090 24 GiB)

| Model | VRAM (typical) | Notes |
|-------|----------------|-------|
| `0.4b` | Lightest | Fast preview |
| `0.8b` | Moderate | Good speed/quality |
| **`1b`** | **Default** | Official demo size; fits 24 GiB with single-pass inference |
| `5b` | Heavy (~24 GiB checkpoint) | Max quality; often OOM on 1080p multi-person without `--stride 2+` |

**VRAM tips (RTX 4090):** flip-test is **off by default** (upstream config enables it but doubles VRAM).
Use `--flip-test` only when you need max accuracy and have headroom. Pass a **single `.mp4`**
or a clean folder — batch scan skips `*_sapiens_overlay.*` and `processed_sapiens_*` subdirs.
Each video runs in an isolated subprocess so a failed run does not poison the next one.

### Terminal progress (v0.3.74)

During inference the terminal prints `>> vaila/vaila_sapiens:` status lines (model load,
video geometry, CSV write) plus a **tqdm frame bar on stderr** in normal CLI mode.
The first 1–2 minutes are often model load before the bar moves — that is expected.
Use `--quiet` to suppress the bar and heartbeat lines (GUI batch sets this automatically).

### Download weights (any model size)

```bash
# Default 1B (setup script also fetches this)
uv run vaila/vaila_sapiens.py --download-weights --model 1b

# Largest 5B checkpoint — accept Meta license on Hugging Face first
uv run vaila/vaila_sapiens.py --download-weights --model 5b

# Then run inference with the same --model flag
uv run vaila/vaila_sapiens.py -i video.mp4 -o out/ --model 5b --stride 1
```

Checkpoints land in `vaila/models/sapiens2/pose/` (e.g. `sapiens2_5b_pose.safetensors`).
DETR detector is shared across all sizes.

## GUI

1. Open vailá → **YOLO + FB** → **Sapiens2 Pose**
2. **Dir…** = batch all videos in a folder (recursive); **File…** = single video
3. Choose output parent directory
4. Model default `1b`; stride `1` = every frame
5. Run

## GUI → CLI mirror

When you click **Run**, the terminal prints a copy-paste command:

```text
>> vaila/vaila_sapiens: Equivalent CLI (copy/paste):
>>   uv run vaila/vaila_sapiens.py -i ... -o ... --model 1b --stride 1 ...
```

The chooser also prints the launcher when you open **Sapiens2 Pose**:

```bash
uv run python -u vaila/vaila_sapiens.py
```

## CLI

```bash
# GUI (no args)
uv run vaila/vaila_sapiens.py

# Dry-run on test clips
uv run vaila/vaila_sapiens.py \
  -i tests/markerless_2d_analysis/ \
  -o /tmp/sapiens_out \
  --model 1b --dry-run

# Full run
uv run vaila/vaila_sapiens.py \
  -i tests/markerless_2d_analysis/camera_01_cube_test_png_196_190_446_dbox_h265.mp4 \
  -o /tmp/sapiens_out \
  --model 1b --stride 1

# Long clips — lower compute
uv run vaila/vaila_sapiens.py -i long.mp4 -o out/ --model 1b --stride 3
```

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `-i` / `--input` | — | Video file or directory |
| `-o` / `--output` | — | Parent output directory |
| `--model` | `1b` | `0.4b`, `0.8b`, `1b`, `5b` |
| `--stride` | `1` | Infer every N frames; overlay reuses nearest pose |
| `--kpt-thr` | `0.3` | Keypoint visibility threshold |
| `--device` | `0` | CUDA device index |
| `--no-overlay` | off | Skip overlay MP4 |
| `--dry-run` | off | Plan only |
| `--download-weights` | off | HF download for selected model + DETR |
| `--quiet` | off | Minimal output (no tqdm / frame heartbeat) |
| `--open-help` | off | Open this help in browser |

## Outputs

### Output directory (v0.3.76)

Each CLI or GUI run creates **one** timestamped folder under the output parent:

```text
<output_parent>/processed_sapiens_YYYYMMDD_HHMMSS/<video_stem>/   ← CSVs, overlay, JSON
```

Subprocess-per-video isolation (default) reuses the parent `--output-base`; isolated
workers write only to `<timestamp>/<stem>/` and no longer mint a second empty
`processed_sapiens_*` directory.

Under `processed_sapiens_YYYYMMDD_HHMMSS/<video_stem>/`:

| File | Description |
|------|-------------|
| `<stem>_sapiens_overlay.mp4` | Skeleton overlay (full frame count) |
| `<stem>_predictions.json` | Per-frame instances (bbox + 308 kp) |
| `<stem>_sapiens_vaila.csv` | Long CSV: `frame,person_id,kpt_idx,x,y,score` (all keypoints) |
| **`<stem>_markers.csv`** | **REC2D/REC3D** — `frame,p1_x,p1_y,...,pN_x,pN_y` (foot anchor) |
| `sapiens_vaila_center.csv` | Same schema as `sam_vaila_center.csv` (bbox center) |
| `sapiens_vaila_bottom.csv` | Bbox bottom-center (foot proxy) |
| `sapiens_vaila_top/left/right.csv` | Other bbox anchors |
| `sapiens_points.csv` | Foot + bbox center + mid-hip per stable `pN` |
| `sapiens_id_map.csv` | `pN, stable_id, n_frames, first_frame, last_frame` |
| `sapiens_tracks.csv` | Legacy bbox tracks (`stable_id`, `x1..y2`) |
| **`sapiens_bbox_tracks.csv`** | **getpixelvideo Load Tracking** — SAM schema (`obj_id`, `x_px`, `w_px`, …) |
| **`<stem>_id_NN_sapiens_pose.csv`** | **getpixelvideo Load** — wide 308-kp pose per person |
| `<stem>_getpixelvideo_pose.csv` | Alias when only one person is tracked |
| `README_sapiens.txt` | Schema summary |
| `FAILED_sapiens.txt` | Error marker on failure |

### getpixelvideo workflow (v0.3.75)

**Full skeleton (308 keypoints):**

1. **Load** → `<stem>_id_01_sapiens_pose.csv` (or `<stem>_getpixelvideo_pose.csv` for single person)
2. Or **Load Tracking CSV** → `<stem>_sapiens_vaila.csv` → enter `person_id` when prompted

**Multi-person bboxes (detection overlay / YOLO detect export):**

1. **Load Tracking CSV** → `sapiens_bbox_tracks.csv` → press **Enter** at anchor prompt (keep boxes)

**REC2D/REC3D** still uses `<stem>_markers.csv` (one foot point per person).

### Biomechanics workflow

1. Run Sapiens2 on each camera view (same `--stride` across views for multi-camera).
2. Feed `<stem>_markers.csv` into **Rec2D** / **Rec3D** with your DLT parameters.
3. Use `<stem>_sapiens_vaila.csv` for joint-level 2D kinematics (308 keypoints).
4. Optional: load wide pose CSVs or `sapiens_bbox_tracks.csv` in **getpixelvideo** for QA/editing.

## vs SAM 3

| | SAM 3 | Sapiens2 Pose |
|---|-------|----------------|
| Task | Text-prompt segmentation | 308-keypoint pose |
| Prompt | Required (`player`, etc.) | None (DETR finds persons) |
| Long video | `max_frames` VRAM cap | `--stride` temporal skip |
| License | HF gated SAM3 | Sapiens2 License |

## References

- [Sapiens2 paper](https://arxiv.org/pdf/2604.21681)
- [GitHub](https://github.com/facebookresearch/sapiens2)
- [HF collection](https://huggingface.co/facebook/sapiens2)
- [Demo Space](https://huggingface.co/spaces/facebook/sapiens2-pose)
