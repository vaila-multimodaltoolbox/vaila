# vaila_sapiens.py

## Module Information

- **Category:** Markerless 2D / Meta (Facebook)
- **File:** `vaila/vaila_sapiens.py`
- **Version:** 0.3.85
- **Updated:** 2026-07-16
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
6. **GPU & advanced** — `--device`, `--pose-batch-size` (GUI pre-fills per-model default), `--flip-test`, overlay on/off, **Draw person IDs**, **Temporal Re-ID** (online linker + OKS + bidirectional, default on)
7. **Run** — terminal prints `>> Equivalent CLI` with `-i` / `-o` and the flags you chose

## GUI → CLI mirror

When you click **Run**, the terminal prints a copy-paste command with **every flag you chose**
(only `-o` for output — CLI creates `processed_sapiens_<timestamp>/` under that parent):

```text
>> vaila/vaila_sapiens: Equivalent CLI (copy/paste):
>>   uv run vaila/vaila_sapiens.py -i ... -o ... \
>>     --model 1b --stride 1 --kpt-thr 0.3 --bbox-thr 0.3 --nms-thr 0.3 --max-persons 8 \
>>     --device 0 --stabilize-ids --quiet ...
>> (CLI creates processed_sapiens_<timestamp>/ under -o)
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
  --stabilize-ids \
  --reid-max-gap 12 \
  --reid-max-dist 180 \
  --reid-min-iou 0.05 \
  --reid-direction-weight 1.0 \
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

# Re-render overlay only (from existing predictions JSON — ~seconds, no re-inference)
uv run vaila/vaila_sapiens.py \
  --rerender-overlay \
  -i /path/to/video.mp4 \
  -o /path/to/processed_sapiens_<ts>/<stem>/
```

### Person IDs — how to read them

Each detected person gets three related identifiers:

| ID | Where | Meaning |
|----|-------|---------|
| `raw_id` | Internal (inference pass) | Per-frame DETR order (1..N); resets every frame |
| **`stable_id`** | `person_id` in CSVs, `obj_id` in `sapiens_bbox_tracks.csv`, `id NN` in overlay | Cross-frame track after **geometric Re-ID** |
| **`pN`** | `markers.csv` columns `p1`, `p2`, … | Stable slot (sorted `stable_id` → `p1`..`pN`) |
| **`#N`** | overlay MP4 tag (same style as SAM3) | `stable_id` (0-based, like SAM3 `obj_id`) |

**Map file:** `sapiens_id_map.csv` — columns `pN, stable_id, n_frames, first_frame, last_frame`.

**Wide pose files:** `<stem>_id_03_sapiens_pose.csv` uses **`stable_id`** (not `pN`) in the filename.

**Audit trail:** `sapiens_reid_links.csv` — `frame,raw_id,temporal_id,stable_id` when bidirectional refine is on (default), or `frame,raw_id,stable_id` with `--no-reid-bidirectional`.

### Temporal identity (v0.3.79)

Sapiens2 builds **native temporal IDs** during inference (not only a post-pass), using the same `GeometricFrameLinker` family as SAM3/YOLO:

| Layer | When | What |
|-------|------|------|
| **Online linker** | During inference (pass 1) | Assigns `temporal_id` frame-by-frame with bbox + velocity + **pose OKS** (COCO-17 subset) + static anchoring |
| **Bidirectional refine** | After inference (default on) | Forward + backward passes; second half prefers backward IDs (same idea as `reid_markers.py`) |
| **Appearance ReID** | Optional (`--appearance-reid`) | OSNet/boxmot merge for long occlusions |

**Pipeline:**

1. **Pass 1 — inference:** DETR + Sapiens2 per frame; online linker assigns `temporal_id` / `stable_id`.
2. **Pass 2 — refine:** bidirectional geometric Re-ID (default **on**); writes `sapiens_reid_links.csv`.
3. **Pass 3 — overlay:** second video read; skeleton + coloured bbox + `#N` tag.

Disable temporal linking with `--no-stabilize-ids`. Disable bidirectional merge with `--no-reid-bidirectional`.

**Overlay tags:** each person shows `#N` (same convention as SAM3). Use `--no-draw-id` to hide tags.

**Sapiens vs SAM3:** SAM3 gets IDs from the video segmentation model first; Sapiens uses per-frame DETR, so vailá adds **pose OKS**, **static-track anchoring**, and **bidirectional** refine to match or beat SAM on crossings when skeleton is visible. Optional `--appearance-reid` helps long occlusions (requires optional `boxmot`).

**Re-render without re-inference:** `--rerender-overlay` rebuilds the MP4 from `*_predictions.json`.

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
| `--stabilize-ids` | **on** | Temporal Re-ID (online + OKS + bidirectional) | Writes `sapiens_reid_links.csv`; stable IDs in all CSVs/overlay |
| `--no-stabilize-ids` | — | Disable temporal Re-ID | Per-frame `raw_id` only |
| `--reid-bidirectional` | **on** | Forward + backward refine merge | `--no-reid-bidirectional` = single forward pass |
| `--reid-max-gap` | `12` | Max frames without detection before track dies | Same as SAM3 / yolov26track |
| `--reid-max-dist` | `180` | Max link-point distance (px) | Hungarian + velocity + OKS linker |
| `--reid-min-iou` | `0.05` | Min bbox IoU gate | Pairs below need proximity match |
| `--reid-direction-weight` | `1.0` | Velocity-direction penalty | Stronger default than SAM3 (`0.5`) — DETR is per-frame |
| `--reid-static-speed` | `5.0` | Static-track speed threshold (px/frame) | Locks low-speed tracks to anchor |
| `--reid-static-radius` | `65` | Static-track anchor radius (px) | Penalizes far jumps for static IDs |
| `--appearance-reid` | off | OSNet appearance merge after geometric Re-ID | Requires optional `boxmot` |
| `--appearance-reid-threshold` | `0.6` | Cosine similarity for appearance merge | Higher = stricter matching |
| `--no-overlay` | off | Skip skeleton preview MP4 | CSV/JSON still written |
| `--no-draw-id` | off | Hide `#N` tags on overlay | Bbox + skeleton only |
| `--rerender-overlay` | off | Rebuild overlay MP4 from `*_predictions.json` | `-i` video + `-o` run dir; no pose re-inference |
| `--dry-run` | off | Plan only | No GPU, no output files |
| `--download-weights` | off | HF download for `--model` + DETR | Writes to `vaila/models/sapiens2/`, then exits |
| `--quiet` | off | Minimal terminal output | No tqdm bar / heartbeat (GUI batch uses this) |
| `--open-help` | off | Open help in browser | Exits after opening `vaila_sapiens.html` |

## Outputs

### Output directory (v0.3.78)

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
| `<stem>_sapiens_overlay.mp4` | Skeleton overlay + per-person `#N` tag (see `sapiens_id_map.csv`) |
| `<stem>_predictions.json` | Per-frame instances (bbox + 308 kp) |
| `<stem>_sapiens_vaila.csv` | Long CSV: `frame,person_id,kpt_idx,x,y,score` (all keypoints) |
| **`<stem>_markers.csv`** | **REC2D/REC3D** — `frame,p1_x,p1_y,...,pN_x,pN_y` (foot anchor) |
| `sapiens_vaila_center.csv` | Same schema as `sam_vaila_center.csv` (bbox center) |
| `sapiens_vaila_bottom.csv` | Bbox bottom-center (foot proxy) |
| `sapiens_vaila_top/left/right.csv` | Other bbox anchors |
| `sapiens_points.csv` | Foot + bbox center + mid-hip per stable `pN` |
| `sapiens_id_map.csv` | `pN, stable_id, n_frames, first_frame, last_frame` |
| **`sapiens_reid_links.csv`** | **Re-ID audit** — `frame,raw_id,stable_id` (like `sam_reid_links.csv`) |
| `sapiens_tracks.csv` | Bbox tracks (`stable_id`, `x1..y2`) |
| **`sapiens_bbox_tracks.csv`** | **getpixelvideo Load Tracking** — SAM schema (`obj_id`, `x_px`, `w_px`, …) |
| **`<stem>_id_NN_sapiens_pose.csv`** | **getpixelvideo Load** — wide 308-kp pose per person (`frame,nose_x,nose_y,left_eye_x,…` Sociopticon names, not `kp000`) |
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
3. Use `<stem>_sapiens_vaila.csv` for joint-level 2D kinematics (308 keypoints; join `kpt_idx` to `keypoint_names` in `<stem>_predictions.json` or Sociopticon labels below).
4. Optional: load wide pose CSVs or `sapiens_bbox_tracks.csv` in **getpixelvideo** for QA/editing.

### Sociopticon keypoint index (kinematics)

Wide pose CSV columns use **anatomical names** from Meta's Sociopticon 308 topology (not `kp000_x`).

| `kpt_idx` | Name | Region |
|-----------|------|--------|
| 0–4 | `nose`, `left_eye`, `right_eye`, `left_ear`, `right_ear` | Head (COCO-like) |
| 5–8 | `left_shoulder`, `right_shoulder`, `left_elbow`, `right_elbow` | Upper limb |
| 9–14 | `left_hip`, `right_hip`, `left_knee`, `right_knee`, `left_ankle`, `right_ankle` | Trunk / leg |
| 15–20 | `left_big_toe`, `left_small_toe`, `left_heel`, `right_big_toe`, `right_small_toe`, `right_heel` | Feet |
| 21–40 | `right_thumb4` … `right_pinky_finger_third_joint` | Right hand (20 kp) |
| 41 | `right_wrist` | Right wrist |
| 42–61 | `left_thumb4` … `left_pinky_finger_third_joint` | Left hand (20 kp) |
| 62 | `left_wrist` | Left wrist |
| 63–68 | `left_olecranon`, `right_olecranon`, cubital fossa, acromion | Extra body |
| 69 | `neck` | Neck |
| 70–305 | Face mesh (`center_of_glabella`, `tip_of_nose`, …) | Face (236 kp) |
| 306–307 | Ear landmarks | Ears |

For full 308-name list, see `sapiens2/sapiens/pose/configs/_base_/keypoints308.py` (`keypoint_id2name` after teeth removal) or `keypoint_names` in `<stem>_predictions.json`.

## vs SAM 3

| | SAM 3 | Sapiens2 Pose |
|---|-------|----------------|
| Task | Text-prompt segmentation | 308-keypoint pose |
| Prompt | Required (`player`, etc.) | None (DETR finds persons) |
| Geometric Re-ID | `--stabilize-ids` → `sam_reid_links.csv` | `--stabilize-ids` (default on) → `sapiens_reid_links.csv` |
| ID on overlay | `#N` on mask/bbox | `#N` on bbox + skeleton |
| Long video | `max_frames` VRAM cap | `--stride` temporal skip |
| License | HF gated SAM3 | Sapiens2 License |

## References

- [Sapiens2 paper](https://arxiv.org/pdf/2604.21681)
- [GitHub](https://github.com/facebookresearch/sapiens2)
- [HF collection](https://huggingface.co/facebook/sapiens2)
- [Demo Space](https://huggingface.co/spaces/facebook/sapiens2-pose)
