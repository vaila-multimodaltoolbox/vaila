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
5. **Detection & keypoint thresholds** — `--bbox-thr`, `--nms-thr`, `--kpt-thr`, `--max-persons`
6. **GPU & advanced** — `--device`, `--pose-batch-size` (empty = auto), `--flip-test`, overlay on/off
7. **Run** — terminal prints full `>> Equivalent CLI` including `--output-base`

## GUI → CLI mirror

When you click **Run**, the terminal prints a copy-paste command with **every flag you chose**:

```text
>> vaila/vaila_sapiens: Equivalent CLI (copy/paste):
>>   uv run vaila/vaila_sapiens.py -i ... -o ... --output-base .../processed_sapiens_<ts>/ \
>>     --model 1b --stride 1 --kpt-thr 0.3 --bbox-thr 0.3 --nms-thr 0.3 --max-persons 8 \
>>     --device 0 --quiet ...
```

The chooser also prints the launcher when you open **Sapiens2 Pose**:

```bash
uv run python -u vaila/vaila_sapiens.py
```

## CLI

### Full inference — all flags

```bash
uv run vaila/vaila_sapiens.py \
  -i /path/to/video.mp4 \
  -o /path/to/output_parent/ \
  --model 1b \
  --stride 1 \
  --device 0 \
  --bbox-thr 0.3 \
  --nms-thr 0.3 \
  --max-persons 8 \
  --kpt-thr 0.3 \
  --pose-batch-size 2 \
  --flip-test \
  --no-overlay \
  --quiet
```

Also: `uv run vaila/vaila_sapiens.py --print-examples` dumps recipes to the terminal.

### Quick recipes

```bash
# GUI (no args)
uv run vaila/vaila_sapiens.py

# Dry-run on test clips
uv run vaila/vaila_sapiens.py \
  -i tests/markerless_2d_analysis/ \
  -o /tmp/sapiens_out \
  --model 1b --dry-run

# Long clips — lower compute
uv run vaila/vaila_sapiens.py -i long.mp4 -o out/ --model 1b --stride 3
```

### Flags — what each option does

#### Input / output

| Flag | Default | What it is | What changes |
|------|---------|------------|--------------|
| `-i` / `--input` | — | One video file or a folder of videos | Folder = batch; skips `*_sapiens_overlay.*` and `processed_sapiens_*` |
| `-o` / `--output` | — | Parent output directory | Creates `processed_sapiens_YYYYMMDD_HHMMSS/<video_stem>/` |

#### Model and inference

| Flag | Default | What it is | What changes |
|------|---------|------------|--------------|
| `--model` | `1b` | Checkpoint size: `0.4b`, `0.8b`, `1b`, `5b` | Larger = better quality, more VRAM. Use same flag for `--download-weights` and run |
| `--stride` | `1` | Infer pose every **N** frames | `1` = every frame. `2+` = fewer GPU passes; skipped frames reuse nearest pose in overlay/CSVs |
| `--device` | `0` | CUDA GPU index | `0` = first NVIDIA card |
| `--flip-test` | off | Left–right flip ensemble in pose head | ~2× VRAM; slightly better accuracy |
| `--pose-batch-size` | auto | Person crops processed per GPU pose pass (per frame) | Auto: 1 (`5b`), 2 (`1b`), 4 (`0.4b`/`0.8b`). See below |

#### What is `--pose-batch-size`?

Sapiens2 in vailá is **top-down** pose — it does not estimate all joints on the full frame at once. Per frame:

1. **DETR detection** finds people and boxes them (`--bbox-thr`, `--nms-thr`, `--max-persons`).
2. **Sapiens2 pose head** crops each person and predicts 308 keypoints on each crop.

`--pose-batch-size` controls step 2: **how many person crops go through the pose network in one GPU forward pass** on the same frame.

**Example:** 12 persons detected, `--pose-batch-size 4` → 3 pose passes (4+4+4). With `--pose-batch-size 1` → 12 passes (slower, less VRAM).

**Auto (default):** omit the flag or leave the GUI field empty — vailá picks by model: `1` for `5b`, `2` for `1b`, `4` for `0.4b`/`0.8b`.

**When to lower:** CUDA **OOM** on crowded frames → try `--pose-batch-size 1`. Also consider lowering `--max-persons` to cap detections.

**Not `--stride`:** stride skips *frames*; pose-batch-size batches *persons within one frame*.

#### Detection thresholds

| Flag | Default | What it is | What changes |
|------|---------|------------|--------------|
| `--bbox-thr` | `0.3` | Min DETR person-detection score | Higher = fewer persons; lower = more false positives |
| `--nms-thr` | `0.3` | Box overlap deduplication | Lower = stricter one-box-per-person |
| `--max-persons` | `8` | Max persons per frame (top scores) | Lower to save VRAM in crowded scenes |

#### `--kpt-thr` — keypoint confidence cutoff

Each joint gets a **confidence score** (0–1) from the pose model. `--kpt-thr` is the minimum
score for a joint to count as **visible** in post-processing. Default `0.3` (MMPose/Sapiens convention).

| Flag | Default | What it is | What changes |
|------|---------|------------|--------------|
| `--kpt-thr` | `0.3` | Per-joint confidence cutoff | Effect depends on output file (see below) |

**Where `--kpt-thr` applies:**

- **Overlay MP4** — joints below threshold are not drawn
- **`<stem>_id_NN_sapiens_pose.csv`** / **`<stem>_getpixelvideo_pose.csv`** — low-confidence joints → empty cells
- **`sapiens_points.csv`** — mid-hip (`pN_hx`, `pN_hy`) averaged only from hip joints above threshold; foot/center use bbox, not this threshold

**Where `--kpt-thr` does *not* filter:**

- **`<stem>_sapiens_vaila.csv`** — long format keeps all joints + raw `score` column
- **`<stem>_markers.csv`** — foot from bbox bottom anchor
- **`sapiens_vaila_*.csv`** — bbox anchors only
- **`<stem>_predictions.json`** — raw scores preserved

**Practical tuning:** `0.1`–`0.2` = more joints (noisier); `0.3` = default; `0.4`–`0.6` = stricter (fewer gaps vs less jitter). Saved in JSON as `kpt_thr_used`.

#### Run control

| Flag | Default | What it is | What changes |
|------|---------|------------|--------------|
| `--no-overlay` | off | Skip skeleton preview MP4 | CSV/JSON still written |
| `--dry-run` | off | Plan only | No GPU, no output files |
| `--download-weights` | off | HF download for `--model` + DETR | Writes to `vaila/models/sapiens2/`, then exits |
| `--quiet` | off | Minimal terminal output | No tqdm bar / heartbeat (GUI batch uses this) |
| `--open-help` | off | Open help in browser | Exits after opening `vaila_sapiens.html` |

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
