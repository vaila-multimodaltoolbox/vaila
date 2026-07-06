# vaila_sam

## Module Information

- **Category:** Multimodal Analysis / Video Segmentation
- **File:** `vaila/vaila_sam.py`
- **Version:** 0.3.71
- **Updated:** 05 July 2026
- **Authors:** Paulo Santiago, Sergio Barroso, Felipe Dias, Lennin Abrão
- **GUI Interface:** Yes (Tkinter batch dialog when no CLI args)
- **CLI Interface:** Yes (`-i`, `-o`, `-t`, ...)
- **Email:** paulosantiago@usp.br
- **GitHub:** [vaila-multimodaltoolbox/vaila](https://github.com/vaila-multimodaltoolbox/vaila)
- **License:** AGPL-3.0

## Description

**vaila_sam** performs video segmentation using **SAM 3** (Segment Anything Model 3) from Meta. It supports text-prompt masks via Hugging Face checkpoints for single or batch video processing.

### Main features:

- Text-prompt video segmentation with SAM 3 / SAM 3.1 Multiplex
- Automatic VRAM management (auto `max_frames` scales with GPU size; `--max-frames 0` = full clip for correct overlay timing)
- Batch processing (directory of videos or single file)
- Overlay MP4 output with coloured masks (temporally nearest SAM keyframe when `--max-frames` subsamples input)
- Per-frame mask PNG output
- Frame-by-frame fallback for low-VRAM GPUs
- GUI mode (Tkinter dialog) and headless CLI mode
- SAM ID stabilization/ReID export (`--stabilize-ids`) with cross-chunk overlap linking, final `sam_tracks.csv`, `sam_reid_links.csv`, `sam_points.csv`, `sam_id_map.csv`, and `*_georeid.csv` aliases for YOLOTrain outputs
- **Cross-Chunk Tracklet Linking** (sliding-window overlap between adjacent chunks, tunable with `--overlap-frames N` default 2; chunk-local SAM IDs are matched into persistent global IDs using bipartite IoU + centroid-distance + optional mask IoU cost matrix solved with Hungarian assignment)
- **Unified geometric Re-ID (v0.3.68)**: `_stabilize_sam_track_ids` now uses `GeometricFrameLinker` from `vaila/geometric_reid.py` (Hungarian + velocity-direction penalty), consistent with the YOLO linker
- **FIFA Skeletal Tracking Light** pipeline via `fifa` subcommand

---

## Installation

### 1. SAM 3 Video (standard)

```bash
uv sync --extra sam
```

### 2. Workstation (NVIDIA CUDA template)

After switching to the Linux/Windows CUDA `pyproject` (see `bin/use_pyproject_*`):

```bash
uv sync --extra gpu --extra sam
```

### 3. FIFA Skeletal Tracking Light (optional)

```bash
uv sync --extra fifa
# With CUDA template:
uv sync --extra gpu --extra fifa --extra sam
```

> **Runtime:** SAM 3 video inference needs an **NVIDIA GPU with CUDA**, even if the `sam` extra is installed on a CPU-only machine.

> **Docs:** Hybrid CPU laptop vs NVIDIA workstation — see [`AGENTS.md`](https://github.com/vaila-multimodaltoolbox/vaila/blob/main/AGENTS.md) in the repository root.

### Platform: CUDA, CPU-only, Windows, macOS

| Environment | SAM 3 *video* in vailá |
|-------------|------------------------|
| Linux / Windows + NVIDIA + CUDA PyTorch | Supported |
| Windows **without** NVIDIA CUDA (CPU / integrated only) | **Not supported** — use Markerless 2D, YOLO, or a remote CUDA machine |
| macOS (Apple Silicon MPS / integrated) | **Not supported** for SAM 3 video — same CUDA requirement; use other vailá GPU workflows where applicable (e.g. Metal-backed stacks), or run SAM 3 on a CUDA host |

`--frame-by-frame` (CLI) / GUI “frame-by-frame” reduces **GPU memory** use on CUDA; it does **not** enable CPU inference.

---

## Weights / Checkpoints

### SAM 3 Video Weights (facebook/sam3)

The Hugging Face repo `facebook/sam3` is **gated**. You must accept the license on [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3), then authenticate:

```bash
uv run hf auth login
uv run vaila/vaila_sam.py --download-weights
```

Or download manually:

```bash
uv run hf download facebook/sam3 sam3.pt --local-dir vaila/models/sam3
```

### Checkpoint Resolution Order

The resolver auto-detects checkpoints in this priority:

1. CLI `-w` / `--checkpoint` argument
2. Environment: `SAM3_CHECKPOINT` or `VAILA_SAM3_CHECKPOINT`
3. `vaila/models/sam3/sam3.pt` (flat layout)
4. `vaila/models/sam3/sam3.1_multiplex.pt` (SAM 3.1)
5. `vaila/models/sam3/sam3_weights/sam3.pt` (nested HF local-dir layout)
6. `models/sam3/sam3.pt` (repo-root models layout)
7. `models/sam3/sam3.1_multiplex.pt` (repo-root models layout)
8. `models/sam3/sam3_weights/sam3.pt` (repo-root nested HF layout)
9. Legacy: `<repo>/sam3_weights/sam3.pt` (deprecated)
10. If none found: auto-download from Hugging Face Hub at runtime

> **SAM 3D Body weights are different!** `vaila/models/sam-3d-dinov3/` is for the FIFA Skeletal pipeline, **not** SAM 3 video. Passing a SAM 3D Body checkpoint to SAM 3 video will be rejected.

---

## SAM 3 Video CLI

### Quick Start

```bash
# Single video — broadcast match
uv run vaila/vaila_sam.py \
  -i tests/sport_fields/ENG_FRA_220243.mp4 \
  -o tests/sport_fields/runs/sam3 \
  -t player --max-frames 80 --max-input-long-edge 1280 \
  --postprocess-points foot

# Batch (all videos in a directory) — auto subprocess-per-video isolation
uv run vaila/vaila_sam.py -i videos_dir/ -o output/ -t player

# Large videos (1080p+ / long clips): stable caps to avoid OOM cascades
# (also useful if your desktop session is still holding VRAM)
uv run vaila/vaila_sam.py -i video.mp4 -o output/ -t person \
  --max-frames 48 --max-input-long-edge 1280 --no-png

# Download weights only
uv run vaila/vaila_sam.py --download-weights

# Open help in browser
uv run vaila/vaila_sam.py --open-help

# Smoke / dry-run (prints effective settings, detected weights, and retry ladder)
uv run vaila/vaila_sam.py -i video.mp4 -o output/ -t person --dry-run
```

### What happens on CUDA OOM (important for big videos)

- The script runs each video in an **isolated subprocess** by default (even for a single video). This prevents SAM3's CUDA state from leaking into subsequent runs.
- If SAM3 exhausts its in-process OOM retry ladder (frame caps and long-edge caps), the per-video subprocess exits and the **coordinator process** automatically runs the **chunked divide-and-conquer** fallback from a clean GPU state.
- Chunked fallback uses a conservative chunk size (≤48 frames), shares 2 overlap frames between adjacent chunks, links chunk-local IDs by same-frame IoU/centroid matching, drops duplicate overlap frames, and regenerates the final overlay from remapped masks so displayed IDs match the final CSVs.

---

## Re-ID and stable track IDs (CLI)

SAM 3 assigns **fresh object IDs inside each temporal chunk**. For sports broadcasts and long clips you usually need **one consistent ID per player** across the full video. vailá applies Re-ID in **two layers** — use both for long chunked runs.

### Layer 1 — Cross-chunk linking (automatic)

When the coordinator falls back to **chunked divide-and-conquer** (OOM, low-FPS guard, or very long clips):

| What | Detail |
|------|--------|
| When | Automatic during `_merge_chunk_outputs` — no extra flag |
| Overlap | `--overlap-frames N` (default **2**) shared frames between adjacent chunks |
| Algorithm | Hungarian assignment on IoU + centroid distance (+ mask IoU when PNGs exist) |
| Output | Merged `sam_tracks.csv`, `sam_frames_meta.csv`, overlay with **global** IDs |

Tune overlap for fast motion or camera cuts:

```bash
uv run vaila/vaila_sam.py -i long_match.mp4 -o out/ -t player \
  --max-frames 128 --max-input-long-edge 1280 \
  --overlap-frames 3
```

### Layer 2 — Geometric ID stabilization (`--stabilize-ids`)

After SAM export (single-pass **or** merged chunk), optionally **rewrite IDs** for short-gap continuity (occlusions, SAM flicker, ID swaps within a chunk):

| What | Detail |
|------|--------|
| Flag | `--stabilize-ids` (CLI: **off** by default; GUI: **on** by default — checkbox *ReID/Stabilize SAM IDs + final CSVs*) |
| Engine | `GeometricFrameLinker` from `vaila/geometric_reid.py` (Hungarian + velocity-direction penalty) |
| Requires | `sam_tracks.csv` (auto-enabled; forces `--save-tracks-csv` if missing) |
| Rewrites | `sam_tracks.csv`, `sam_frames_meta.csv` |
| Extra file | `sam_reid_links.csv` — columns `frame,old_obj_id,obj_id` (audit trail) |
| Post-process | When `sam_reid_links.csv` exists, batch post-process also writes `sam_points_georeid.csv` and `sam_id_map_georeid.csv` (aliases for YOLOTrain / getpixelvideo) |

**Recommended CLI for tracking workflows** (short clip, single athlete, or lab test):

```bash
uv run vaila/vaila_sam.py \
  -i /path/to/clip.mp4 \
  -o /path/to/output_parent/ \
  -t person \
  --stabilize-ids \
  --postprocess-points all
```

**Long broadcast with chunking + Re-ID** (your ~6k-frame case):

```bash
uv run vaila/vaila_sam.py \
  -i /path/to/long_match.mp4 \
  -o /path/to/output_parent/ \
  -t person \
  --max-frames 128 \
  --max-input-long-edge 1280 \
  --stabilize-ids \
  --overlap-frames 2 \
  --postprocess-points all
```

Log lines to expect:

```text
[SAM3-CHUNK] Merging 147/147 chunk results...
[SAM3-ReID] geometric ID stabilization: 42 track(s) -> sam_reid_links.csv
[postprocess] wrote 1 sam_points.csv + 5 vailá anchor CSV(s).
```

### Re-ID output files (quick reference)

| File | Role |
|------|------|
| `sam_tracks.csv` | Long format: bbox + centroids per frame/object (pixels) — **load in getpixelvideo** |
| `sam_bbox_tracks.csv` | Hardlink/copy alias of `sam_tracks.csv` (easier to spot) |
| `sam_reid_links.csv` | Frame-level ID remap log (only with `--stabilize-ids`) |
| `sam_points.csv` | Wide marker CSV for getpixelvideo / rec2d (foot canonical by default) |
| `sam_id_map.csv` | Maps `pN` columns → SAM `obj_id` |
| `sam_points_georeid.csv` | Copy of `sam_points.csv` when Re-ID ran (YOLOTrain convention) |
| `sam_vaila_bottom.csv` | Simple `frame,x1,y1,…` foot anchors — direct rec2d / field homography |

### Downstream: load tracks in getpixelvideo

1. Open the video in **getpixelvideo** (Frame C).
2. **Load Tracking CSV** → pick `sam_tracks.csv` or `sam_bbox_tracks.csv`.
3. When prompted for anchor: `2` = bottom (foot), `1` = center, etc.
4. **Save** writes editable markers; or keep overlay-only with Enter.

For stabilized IDs prefer `sam_points.csv` or `sam_vaila_bottom.csv` after `--postprocess-points all`.

### Re-run post-process only (no GPU)

If SAM finished but post-process failed (e.g. older build + `.avi` overlay):

```bash
uv run vaila/sam_postprocess.py \
  /path/to/processed_sam_…/video_stem \
  --mode all
```

Writes `sam_points.csv`, `sam_id_map.csv`, and five `sam_vaila_*.csv` files.

---

### Common errors (CLI **and** GUI)

| Symptom in the terminal / GUI log | Most likely cause | Fix |
|-----------------------------------|-------------------|-----|
| `Could not open VideoWriter for SAM3 subsample` in `FAILED_sam.txt` | OpenCV still could not create the temporal subsample clip (`_sam3_subsample_input.*`) after the mp4v → MJPG/XVID → ffmpeg libx264 pipe fallback. v0.3.71+ skips broken `h264_v4l2m2m`/`avc1` on ARM boards; v0.3.69+ routes extreme low-FPS subsamples to chunked fallback before opening the writer. | Delete the failed per-video output folder and re-run. Ensure `ffmpeg` is on `PATH` for the pipe fallback. For normal long broadcasts prefer auto `max_frames` or `--max-frames 128`/`256`; if the writer failure repeats, re-encode the source video. |
| `[h264_v4l2m2m] Could not find a valid device` (stderr noise) | Harmless on many Linux SBCs when OpenCV probes the hardware encoder before falling back to `mp4v`. v0.3.71 tries software codecs first and uses ffmpeg libx264 when OpenCV fails entirely. CSV/JSON exports continue even if overlay MP4 stitching fails. | Ignore if the run completes. Install `ffmpeg` if overlay MP4 is missing. Chunked runs now auto-generate `sam_points.csv` + `sam_vaila_*.csv` after merge. |
| `ERROR on <video>: subprocess killed by SIGKILL (exit=-9). Likely the Linux OOM killer (SYSTEM RAM, not VRAM)…` | **Host RAM** OOM killer — the temporal subsample plus SAM3's CPU-side video tensor demanded more system RAM than the OS could provide. Long broadcast clips (16 k+ frames @ 1080 p) easily peak >30 GiB on host RAM. | Lower `--max-frames` (try 256 → 128 → 64), add `--max-input-long-edge 1280`, close other heavy apps, and re-run. Confirm with `dmesg \| tail` / `journalctl -k -n 50`. |
| `subprocess killed by SIGSEGV (exit=-11)` | Native segfault inside CUDA / Torch / OpenCV. | Check `nvidia-smi`, update GPU driver, rerun with `--dry-run` / `--preflight` to isolate. |
| `subprocess killed by SIGABRT (exit=-6)` | C++ abort from Torch / Triton destructor running on the wrong thread. | Reproduce from the CLI (no Tk), attach `gdb` to the child if persistent. |
| `subprocess killed by SIGBUS (exit=-7)` | Corrupted video file or broken mmap. | Re-encode the input with `ffmpeg -i in.mp4 -c:v libx264 -c:a copy out.mp4`. |
| `CUDA out of memory` (Python exception, GPU side) | VRAM OOM during inference. | Lower `--max-input-long-edge` (1280/960), or add `--frame-by-frame` (loses temporal tracking). |
| `SAM3_NEEDS_CHUNKING` or `subprocess exit=7 (EXIT_NEEDS_CHUNKING)` | Per-video child exhausted its OOM ladder, or an extreme temporal subsample would require an impractically low temp-video FPS (for example `max_frames=1` on a long broadcast). | The coordinator already retries with the **chunked divide-and-conquer** fallback automatically. For normal long clips prefer auto `max_frames` or `--max-frames 128`/`256`; keep `1` for short diagnostics only. |
| `[sam_postprocess] FAILED … Cannot determine frame size` after a **chunked** run that wrote `*_sam_overlay.avi` | Batch post-process only looked for `*_sam_overlay.mp4` and `source_original=` in `README_sam.txt`; chunked merges often emit MJPG `.avi` overlays and `source=` in the README. v0.3.69+ also reads AVI overlays, `sam_contours.json` width/height, and `source=` / `source_original=`. | Re-run post-process only (no SAM re-inference): `uv run vaila/sam_postprocess.py -i /path/to/processed_sam_…/video_stem --mode all`. Or upgrade and re-run the GUI — the fix applies on the next batch. |
| GUI dialog opens but the SAM run fails immediately with `[GUI] subprocess exited with code 0` and `Failed (1/1)` | Worker subprocess died **after** SAM3 model load but before producing any output (frequently SIGKILL by the OOM killer). The GUI now also prints `[GUI] subprocess killed by SIGKILL …` and the path to the full log. | Same as `SIGKILL` row above — lower `--max-frames`. |

> **Pro tip:** the CLI now prints a runtime banner with **GPU VRAM** and **Host RAM** numbers at startup; combine that with `--print-examples` to pick the right `--max-frames` for your machine.

### CLI workflows by scenario

Pick the recipe that matches your clip. All paths assume CUDA + `uv sync --extra sam`.

| Scenario | Goal | Key flags |
|----------|------|-----------|
| **A — Lab / short clip** | Quick segmentation + markers | auto `max_frames`, `--postprocess-points all` |
| **B — Single athlete** | Stable ID on a sprint/drill (~100 frames) | `--stabilize-ids` |
| **C — Long broadcast** | Full match without host-RAM OOM | `--max-frames 128` + `--max-input-long-edge 1280` |
| **D — Long + Re-ID** | Consistent player IDs after chunking | C + `--stabilize-ids` + `--overlap-frames 2` |
| **E — Batch folder** | Many clips, isolated subprocess each | `-i clips_dir/` (isolation on by default) |
| **F — Low VRAM (8 GiB)** | Never OOM; per-frame masks only | `--frame-by-frame --no-overlay` |
| **G — Tracks-only / rec2d** | CSV bboxes, minimal disk | `--tracks-only --postprocess-points foot` |
| **H — Post-process repair** | SAM done; CSVs missing | `uv run vaila/sam_postprocess.py …` |

```bash
# Print all recipes again (no GPU):
uv run vaila/vaila_sam.py --print-examples

# Open this help page:
uv run vaila/vaila_sam.py --open-help

# --- A: Short clip, 24 GiB GPU (auto caps) ---
uv run vaila/vaila_sam.py -i clip.mp4 -o out_parent/ -t player \
  --postprocess-points all

# --- B: Single athlete / drill (Re-ID on) ---
uv run vaila/vaila_sam.py -i sprint.mp4 -o out_parent/ -t person \
  --stabilize-ids --postprocess-points all

# --- C: Long broadcast (~15k+ frames) — avoid SIGKILL / host-RAM OOM ---
uv run vaila/vaila_sam.py -i long_match.mp4 -o out_parent/ -t player \
  --max-frames 128 --max-input-long-edge 1280 --postprocess-points all

# --- D: Long broadcast + stable IDs (chunk merge + geometric Re-ID) ---
uv run vaila/vaila_sam.py -i long_match.mp4 -o out_parent/ -t person \
  --max-frames 128 --max-input-long-edge 1280 \
  --stabilize-ids --overlap-frames 2 --postprocess-points all

# --- E: Batch directory (one subprocess per video by default) ---
uv run vaila/vaila_sam.py -i clips_dir/ -o out_parent/ -t player \
  --max-frames 256 --stabilize-ids --postprocess-points foot

# --- F: Low-VRAM GPU — no temporal tracking ---
uv run vaila/vaila_sam.py -i video.mp4 -o out_parent/ -t person \
  --frame-by-frame --no-png --no-overlay

# --- G: Fast bbox export for rec2d / homography (no overlay video) ---
uv run vaila/vaila_sam.py -i match.mp4 -o out_parent/ -t player \
  --tracks-only --stabilize-ids --postprocess-points foot

# --- H: Repair post-process on an existing SAM run (no re-inference) ---
uv run vaila/sam_postprocess.py out_parent/processed_sam_…/video_stem --mode all

# --- Utilities ---
uv run vaila/vaila_sam.py -i video.mp4 -o out/ -t person --dry-run   # smoke / caps
uv run vaila/vaila_sam.py -i clips_dir/ --preflight -o out/          # SAM3_PREFLIGHT.csv
uv run vaila/vaila_sam.py --download-weights                        # HF sam3.pt
uv run vaila/vaila_sam.py -i video.mp4 -o out/ -t player \
  -w vaila/models/sam3/sam3.1_multiplex.pt                           # SAM 3.1 weights
```

> Equivalent GUI: launch `vaila.py`, **Frame B → Video AI tools → SAM (Segment Anything)**, fill **Input**, **Output**, **Text prompt**, **Max frames** and click **Run**. The progress window now mirrors every CLI banner and prints the same exit-code diagnosis.

### Companion AI seed for the soccer field

Pair SAM 3 with the bundled YOLO-pose detector for the 32 pitch
keypoints:

```bash
uv run python -m vaila.soccerfield_keypoints_ai \
  --mode video \
  -i tests/sport_fields/ENG_FRA_220243.mp4 \
  -o tests/sport_fields/runs/pitch_kps \
  --backend ultralytics \
  --weights vaila/models/runs/pose_fifa/pitch32_recipeA_400ep/weights/best.pt \
  --imgsz 1280 --conf 0.30 --draw-min-conf 0.40 \
  --device 0 --stride 5 --max-frames 60 --overlay-video
```

> The bundled detector (`pitch32_recipeA_400ep`) reaches **Pose mAP50 = 0.945**
> on the public `football-field-detection-f07vi` split (April 2026).
> See [`soccerfield_keypoints_ai.html`](soccerfield_keypoints_ai.html).

### CLI Flags

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--input` | `-i` | Path | — | Input video file or directory of videos (batch) |
| `--output` | `-o` | Path | — | Output base directory |
| `--text` | `-t` | str | `person` | Open-vocabulary text prompt for segmentation (no fixed class list) |
| `--frame` | `-f` | int | `0` | Frame index used for the initial prompt |
| `--weights` / `--checkpoint` | `-w` | Path | auto | SAM 3 checkpoint (file or folder); auto-detected if omitted |
| `--no-overlay` | — | flag | — | Skip overlay MP4 output |
| `--no-png` | — | flag | — | Skip mask PNG output |
| `--tracks-only` | — | flag | — | Fast profile: skip overlay, PNG masks and contours; write bbox/centroid CSV only |
| `--delete-mask-png` | — | flag | — | Delete bulky `masks/` and `sam_masks_manifest.csv` after exports finish (this is now the default behavior) |
| `--keep-mask-png` / `--keep-masks` | — | flag | — | Keep `masks/` and `sam_masks_manifest.csv` after exports finish (by default they are deleted to save disk space) |
| `--stabilize-ids` | GUI: `ReID/Stabilize SAM IDs + final CSVs` | flag | off in CLI; **on** in GUI | After export, rewrite SAM IDs with geometric continuity (IoU/centroid/velocity). Writes `sam_reid_links.csv`, updates `sam_tracks.csv` + `sam_frames_meta.csv`. Post-process then emits `sam_points_georeid.csv` aliases. **Recommended for tracking / rec2d workflows.** |
| `--overlap-frames` | — | int | `2` | Shared frames between adjacent chunks for cross-chunk ID linking (chunked fallback only). Increase to `3`–`4` for fast motion. |
| `--chunk-size` | — | int | auto (≤48) | Chunk length for divide-and-conquer fallback. Larger = fewer model reloads but more VRAM per chunk. |
| `--[no-]overlay-rich` | — | bool | `true` | Enrich overlay with bbox/ID/score/contours (on top of the colored masks) |
| `--[no-]draw-contour` | — | bool | `true` | Draw mask contours on the overlay |
| `--[no-]draw-box` | — | bool | `true` | Draw bounding boxes on the overlay |
| `--[no-]draw-id` | — | bool | `true` | Draw `#obj_id score` label on the overlay |
| `--draw-centroid` | — | flag | — | Draw the mask centroid on the overlay |
| `--[no-]save-contours` | — | bool | `true` | Write `sam_contours.json` (polygons per frame/object in pixels) |
| `--[no-]save-tracks-csv` | — | bool | `true` | Write `sam_tracks.csv` (long format, bbox, area, polygon stats and mask centroid `cx_px/cy_px`) |
| `--contours-format` | — | choice | `json` | `json` or `jsonl` (`sam_contours.json` vs `sam_contours.jsonl`) |
| `--contours-gzip` | — | flag | — | Write gzipped contours output (`.json.gz` / `.jsonl.gz`) |
| `--frame-by-frame` | — | flag | — | Process each frame individually (prevents OOM but loses temporal tracking) |
| `--max-frames` | — | int | auto | Max frames passed to SAM3 on GPU (VRAM cap); `0` = full clip |
| `--max-input-long-edge` | — | int | auto | Max long edge (px) for frames fed to SAM3; `0` = native resolution; try `1280`/`960` for 4K+ or OOM |
| `--dry-run` / `--smoke` | — | flag | — | Print effective settings/caps/checkpoint and exit (does not run SAM3) |
| `--postprocess-points` | — | choice | `all` | Build per-video vailá-format pixel CSVs (`sam_points.csv` + five `sam_vaila_*.csv` anchor files) after batch: `foot` (bottom-center of bbox, best for soccer-field homography + `rec2d`), `center` (bbox center), `mask` (mask centroid from PNG or `sam_tracks.csv`), `all` (foot + extra cx/cy/mx/my columns), `none` (skip post-processing) |
| `--download-weights` | — | flag | — | Download facebook/sam3 into `vaila/models/sam3/` |
| `--open-help` | — | flag | — | Open help page in the browser |
| `--no-isolate-batch` | — | flag | — | Disable subprocess-per-video isolation (not recommended; can cause cascading OOM after a failure). Default behavior is isolation-enabled for both single-video and batch runs. |
| `--video-output-dir` | — | Path | — | **Internal/hidden**: used by subprocess-per-video isolation; write outputs directly to this directory (single-video mode only) |
| `--no-chunked-fallback` | — | flag | — | **Internal/hidden**: recursion guard used by coordinator/chunk subprocesses (prevents infinite `_chunks/out_*/_chunks/...` on repeated OOM) |

### CLI — full command (all options)

Use this as a **copy/paste template** showing every CLI option that exists for SAM 3 video:

```bash
uv run vaila/vaila_sam.py \
  --input "/path/to/video_or_dir" \
  --output "/path/to/output_parent" \
  --text "person" \
  --frame 0 \
  --checkpoint "/path/to/sam3.pt" \
  --max-frames 80 \
  --max-input-long-edge 1280 \
  --overlay-rich \
  --draw-contour \
  --draw-box \
  --draw-id \
  --save-contours \
  --save-tracks-csv \
  --contours-format json \
  --postprocess-points all \
  --dry-run \
  --open-help \
  --download-weights \
  --no-overlay \
  --no-png \
  --frame-by-frame \
  --contours-gzip \
  --no-isolate-batch \
  --video-output-dir "/path/to/output_single_video_dir"
```

Notes:

- **Do not** combine `--download-weights` or `--open-help` with a real run: they **exit early**.
- `--dry-run` (alias `--smoke`) prints the effective settings and exits (no GPU work).
- `--video-output-dir` is **internal** (hidden from `--help`) and only valid when `--input` is a **single file**.
- For normal batch runs, **omit** `--no-isolate-batch` (keep isolation enabled by default).

### Invocation Modes

| Mode | How |
|------|-----|
| CLI batch | `uv run vaila/vaila_sam.py -i INPUT -o OUTPUT [-t TEXT]` |
| GUI (Tkinter dialog) | `uv run vaila/vaila_sam.py` (no args) |
| FIFA pipeline | `uv run vaila/vaila_sam.py fifa <subcommand>` |
| Download only | `uv run vaila/vaila_sam.py --download-weights` |

### Output Structure

```
output/
  processed_sam_YYYYMMDD_HHMMSS/
    video_name/
      masks/           (PNG per frame, unless --no-png)
      <video_stem>_sam_overlay.mp4   (unless --no-overlay)
      sam_frames_meta.csv
      sam_contours.json             (unless --no-save-contours)
      sam_tracks.csv                (bbox + centroid, unless --no-save-tracks-csv)
      sam_bbox_tracks.csv           (hardlink/copy of sam_tracks.csv; discoverable alias)
      sam_masks_manifest.csv        (written when masks are saved; index of mask PNGs)
      README_sam.txt                (verbose: every file explained — schema, units, role)
```

Since **v0.3.55**, every SAM3 run writes:

- A **verbose `README_sam.txt`** that documents every file produced in the
  output directory: schema, units, role in the downstream pipeline (vailá
  pixel tool, rec2d/rec3d, ReID, …). Open it first when you come back to an
  old run.
- **`sam_bbox_tracks.csv`** — a sibling **hardlink** (or copy, if the
  filesystem doesn't allow hardlinking) of `sam_tracks.csv`. Same bytes,
  same inode on POSIX; the extra name makes the bbox file easy to spot in a
  long directory listing. `getpixelvideo.py`'s smart loader accepts either
  name (detection is column-based, not filename-based).

### Additional outputs: `sam_points.csv` + vailá anchor CSVs (default)

By default (`--postprocess-points all`), *vailá* post-processes each per-video output directory and writes **pixel CSVs** that can be loaded directly by `getpixelvideo.py` and `rec2d.py`:

```
output/processed_sam_YYYYMMDD_HHMMSS/
  video_name/
    sam_points.csv
    sam_id_map.csv
    sam_vaila_center.csv        # simple frame,x1,y1,...,xN,yN — bbox center
    sam_vaila_bottom.csv        # simple frame,x1,y1,...,xN,yN — bottom-center (foot)
    sam_vaila_top.csv           # simple frame,x1,y1,...,xN,yN — top-center
    sam_vaila_left.csv          # simple frame,x1,y1,...,xN,yN — left-center
    sam_vaila_right.csv         # simple frame,x1,y1,...,xN,yN — right-center
    sam_points_georeid.csv      # written when --stabilize-ids/ReID is active
    sam_id_map_georeid.csv      # written when --stabilize-ids/ReID is active
```

#### vailá anchor CSVs (`sam_vaila_*.csv`)

These five files use a simple format with one (x, y) pair per tracked object:

```
frame,x1,y1,x2,y2,...,xN,yN
0,960.5,540.0,1200.3,480.2,...
1,961.1,541.0,1201.0,481.1,...
```

Each file places the anchor at a different bbox point:
- **center** — bbox center
- **bottom** — bottom-center of bbox (foot point, best for field homography)
- **top** — top-center of bbox (head)
- **left** — left-center of bbox
- **right** — right-center of bbox

Empty cells indicate the object was not detected in that frame.

#### `sam_points.csv` schema

- **Always** starts with `frame`
- Then one block per tracked object in the video: `p1_*`, `p2_*`, … (where `pN` is assigned from
  the **sorted SAM `obj_id` list**; see `sam_id_map.csv`)

Modes:

- `--postprocess-points foot`:
  - Columns: `frame, p1_x, p1_y, p2_x, p2_y, ...`
  - Meaning: `pN_x/pN_y` = **bottom-center of bbox** (pixels)
- `--postprocess-points center`:
  - Columns: `frame, p1_x, p1_y, p2_x, p2_y, ...`
  - Meaning: `pN_x/pN_y` = **bbox center** (pixels)
- `--postprocess-points mask`:
  - Columns: `frame, p1_x, p1_y, p2_x, p2_y, ...`
  - Meaning: `pN_x/pN_y` = **mask centroid** from `masks/frame_*.png` (pixels)
- `--postprocess-points all`:
  - Columns: canonical `pN_x/pN_y` (default canonical is **foot**) plus extras:
    `pN_cx, pN_cy` (bbox center) and `pN_mx, pN_my` (mask centroid)
  - Example header:
    `frame,p1_x,p1_y,p1_cx,p1_cy,p1_mx,p1_my,p2_x,p2_y,...`

Notes:

- Missing detections are written as **empty cells** (CSV blanks).
- The coordinates are computed in **original video pixels**, using the overlay video (MP4 or AVI), `sam_contours.json`, masks, or README `source`/`source_original` to recover frame width/height.

#### `sam_id_map.csv` schema

Maps `pN` columns to SAM object IDs and their lifespan:

`pN,obj_id,n_frames,first_frame,last_frame`

---

## VRAM Management

SAM 3 loads **all session frames** onto the GPU. Long clips can exceed VRAM.

| Strategy | How |
|----------|-----|
| Auto (default) | vailá picks a frame cap from the **currently free** GPU VRAM (see `[SAM3 VRAM]` in terminal) |
| Environment variable | `SAM3_MAX_FRAMES=80` |
| CLI override | `--max-frames 80` |
| Full clip | `--max-frames 0` (large VRAM only) |
| Frame-by-frame fallback | `--frame-by-frame` (loses temporal tracking) |

> **PYTORCH_CUDA_ALLOC_CONF:** vailá auto-sets `expandable_segments:True` to reduce CUDA OOM from memory fragmentation.

### Batch without VRAM blow-ups (recommended)

For a directory of videos, prefer the **default batch isolation** (one subprocess per video)
and set an explicit frame cap:

```bash
uv run vaila/vaila_sam.py \
  --input "/caminho/para/dir_com_videos" \
  --output "/caminho/para/saida" \
  --text "person" \
  --frame 0 \
  --max-frames 80 \
  --postprocess-points all
```

If you still see CUDA OOM on some clips:

- Lower the cap: `--max-frames 64` or `--max-frames 32`
- Or use frame-by-frame as a last resort (trades temporal tracking for memory safety):

```bash
uv run vaila/vaila_sam.py \
  --input "/caminho/para/dir_com_videos" \
  --output "/caminho/para/saida" \
  --text "person" \
  --frame 0 \
  --frame-by-frame \
  --postprocess-points all
```

Important:

- Keep **subprocess-per-video isolation enabled** in batch mode (default). Avoid `--no-isolate-batch`
  unless debugging: a CUDA OOM inside SAM3 can leak GPU state and poison the next videos.

### Batch behaviour (CLI and GUI)

- Between videos vailá **drops the predictor** and runs `gc.collect()` + `torch.cuda.empty_cache()` so video #N+1 starts from a clean GPU state.
- If a video still hits **CUDA OOM**, vailá **auto-retries** the same clip with `max_frames=64` and then `max_frames=32` before failing.
- A failed video keeps a small **`FAILED_sam.txt`** in its output folder describing the reason — so empty/missing dirs are no longer silent.
- The GUI shows a **progress window** with per-video status, a scrollable log and a **Cancel** button.

---

## Text Prompt Classes

SAM 3 in this module uses **open-vocabulary text prompts**. There is no fixed closed class list beyond `person`.

### Prompt presets (GUI combobox)

The GUI dialog exposes the following presets in the **Text prompt** combobox. The combobox is editable (`state="normal"`), so you can also type any free-text prompt — these are only suggestions:

- `person`, `player`, `goalkeeper`, `referee`, `coach`
- `ball`, `soccer ball`, `basketball`, `volleyball`
- `crowd`, `car`, `bike`, `dog`, `cat`

From the CLI, pass your prompt with `-t "…"`. Example:

```bash
uv run vaila/vaila_sam.py -i match.mp4 -o out/ -t "goalkeeper" -f 0
```

### Prompt tips

- Use short nouns (1-3 words), lowercase. Commas or `.` are ignored.
- If a generic prompt fails, try a **more specific synonym** (`player` instead of `person`; `soccer ball` instead of `ball`) or use a **scene-specific role** (`goalkeeper`, `referee`, `coach`).
- Sport-specific scenarios tested in vailá:
  - Soccer broadcasts → `player`, `goalkeeper`, `referee`, `soccer ball`
  - Basketball → `player`, `basketball`, `referee`
  - Volleyball → `player`, `volleyball`
- Keep the **prompt frame** on a clean, well-exposed moment (no occlusion, steady lens). Use `-f` (CLI) or the "Prompt frame index" field (GUI) to change it.

---

## GUI overview

When you launch `vaila_sam` without CLI args (or from the vailá main window), two Tk windows may appear:

In the main vailá window, open it via:

- **Frame B → "Video AI tools" → "SAM (Segment Anything)"**

1. **`SamVideoDialog`** — configuration modal:
    - `Input (dir or file)` + `Output folder` + `sam3.pt / weights (-w)` browsers.
    - **Text prompt combobox** with the presets listed above (editable).
    - `Prompt frame index`, `Save overlay MP4`, `Save mask PNGs`, **ReID/Stabilize SAM IDs** (on by default), optional frame-by-frame CUDA fallback.
    - Buttons: `Run`, `Cancel`, **`Help`** (opens this page in your default browser).

2. **`SamBatchProgress`** — per-video status window (batch only):

```text
+-----------------------------------------------------------+
| SAM 3 - batch progress                                   |
+-----------------------------------------------------------+
| 2 / 24 processed                                          |
| [==========--------------------------------------------]  |
| Log:                                                      |
|  [14:21:03] Starting video 3/24: ARG_CRO_000737.mp4       |
|  [postprocess] wrote 24 sam_points.csv + 120 vailá ...    |
|  ...                                                      |
|  [Field calibration (after batch finishes)]               |
|    [Calibrate field (DLT2D)]                              |
|  [Request cancel]  [Help]            [Close (disabled)]   |
+-----------------------------------------------------------+
```

- When **Post-process points** is `all` (default), `sam_points.csv`, `sam_id_map.csv`, and the five `sam_vaila_*.csv` anchor files are written automatically at the end of the batch — look for `[postprocess]` lines in the log.
- **ReID/Stabilize SAM IDs** is checked by default in the GUI (equivalent to `--stabilize-ids` on the CLI). Uncheck only for quick mask previews.
- **Calibrate field (DLT2D)** enables after the batch finishes (optional soccer-field homography).

---

## FIFA Skeletal Tracking Light

The `fifa` subcommand delegates to `vaila/fifa_skeletal_pipeline.py`, providing a complete pipeline for the [FIFA Skeletal Tracking Light 2026](https://inside.fifa.com/innovation/innovation-programme/skeletal-tracking) challenge: monocular broadcast video to 3D human pose estimation (15 keypoints).

### Challenge Links

- [Starter Kit (GitHub)](https://github.com/FIFA-Skeletal-Light-Tracking-Challenge/FIFA-Skeletal-Tracking-Starter-Kit-2026)
- [Dataset (Hugging Face)](https://huggingface.co/datasets/tijiang13/FIFA-Skeletal-Tracking-Light-2026)
- [Validation Portal (Codabench)](https://codabench.org/competitions/11681/)
- [Test Portal (Codabench)](https://www.codabench.org/competitions/11682/)
- [Kaggle Mirror](https://www.kaggle.com/competitions/fifa-skeletal-light/overview)
- [WorldPose Dataset (paper)](https://eth-ait.github.io/WorldPoseDataset/)

### FIFA Setup

```bash
# 1. Switch to CUDA pyproject template (workstation)
bash bin/use_pyproject_linux_cuda.sh

# 2. Install extras
uv sync --extra gpu --extra fifa --extra sam

# 3. Clone sam_3d_body + download gated SAM 3D Body weights
#    (accept the license on https://huggingface.co/facebook/sam-3d-body-dinov3 first,
#     then `uv run hf auth login`).
bash bin/setup_fifa_sam3d.sh                # Linux / macOS
# pwsh bin/setup_fifa_sam3d.ps1            # Windows

# 4. Download challenge data (cameras, boxes, skel_2d, skel_3d, etc.)
#    Accept access at: https://huggingface.co/datasets/tijiang13/FIFA-Skeletal-Tracking-Light-2026
#    Videos come separately from FIFA (not on HF)

# 5. Bootstrap the data layout (symlink videos + sequences_*.txt + pitch_points.txt)
uv run vaila/vaila_sam.py fifa bootstrap \
  --videos-dir /path/to/FIFA_Challenge_2026_Video_Data/Videos \
  --data-root  /path/to/FIFA/data
```

### Data Layout

```
data/
  cameras/
    MATCH_CLIP.npz          (camera intrinsics + extrinsics per sequence)
  images/
    MATCH_CLIP/
      00001.jpg ...         (frames extracted from videos)
  videos/
    MATCH_CLIP.mp4          (broadcast video per sequence)
  boxes/
    MATCH_CLIP.npy          (bounding boxes per frame per person)
  skel_2d/
    MATCH_CLIP.npy          (2D keypoints, 15 joints)
  skel_3d/
    MATCH_CLIP.npy          (3D keypoints, 15 joints)
  pitch_points.txt          (FIFA pitch calibration points)
  sequences_full.txt
  sequences_val.txt
  sequences_test.txt
```

### FIFA Subcommands

#### 0. `bootstrap` — symlink videos + sequences_*.txt + pitch_points.txt

| Argument | Required | Description |
|----------|----------|-------------|
| `--videos-dir` | Yes | Source folder with raw `*.mp4` |
| `--data-root` | Yes | Target FIFA data root |
| `--val-sequences` | No | Text file with official val split |
| `--test-sequences` | No | Text file with official test split |
| `--no-copy-fallback` | No | Fail instead of copying when symlinks are denied |
| `--overwrite-pitch-points` | No | Replace any existing `pitch_points.txt` |

```bash
uv run vaila/vaila_sam.py fifa bootstrap \
  --videos-dir /data/FIFA/FIFA_Challenge_2026_Video_Data/Videos \
  --data-root  /data/FIFA/data
```

Run this *before* `fifa prepare` when your videos are not yet copied into
`data/videos/` and when the official splits have not been written.

#### 1. `prepare` — Videos to data/videos + data/images

| Argument | Required | Description |
|----------|----------|-------------|
| `--video-source` | Yes | Directory with source video files |
| `--data-root` | Yes | Target data/ directory |
| `--sequences-out` | No | Write sequences list file |
| `--extract-fps` | No | FPS for frame extraction |

```bash
uv run vaila/vaila_sam.py fifa prepare \
  --video-source /path/to/raw_videos \
  --data-root data/
```

> `prepare` does **not** create `cameras/`, `pitch_points.txt`, or official `boxes/` — those come from the starter kit / HF dataset.

#### 2. `boxes` — Generate bounding boxes

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--data-root` | Yes | — | Data directory |
| `--sequences` | Yes | — | Sequences file (.txt) |
| `--source` | No | `yolo` | `yolo` or `sam3` |
| `--yolo-model` | No | `yolo11n.pt` | YOLO model name |
| `--sam3-checkpoint` | No | auto | SAM3 checkpoint for `--source sam3` |
| `--text-prompt` | No | `person` | Text prompt for SAM3 |
| `--max-persons` | No | `25` | Max persons per frame |

```bash
uv run vaila/vaila_sam.py fifa boxes \
  --data-root data/ \
  --sequences data/sequences_val.txt
```

#### 3. `preprocess` — SAM 3D Body to skel_2d / skel_3d

| Argument | Required | Description |
|----------|----------|-------------|
| `--data-root` | Yes | Data directory |
| `--sequences` | Yes | Sequences file (.txt) |
| `--no-skip` | No | Re-process even if skel_2d/skel_3d already exist |

```bash
uv run vaila/vaila_sam.py fifa preprocess \
  --data-root data/ \
  --sequences data/sequences_val.txt
```

#### 4. `baseline` — Camera tracker + 3D pose to submission NPZ

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--data-root` | Yes | — | Data directory |
| `--sequences` | Yes | — | Sequences file (.txt) |
| `--output` | Yes | — | Output NPZ path |
| `--refine-interval` | No | `1` | Pitch-point refinement interval |
| `--export-camera` | No | — | Save per-frame calibrated cameras |
| `--visualize` | No | — | Enable visualization |
| `--calibration-dir` | No | auto | Camera export directory |

```bash
uv run vaila/vaila_sam.py fifa baseline \
  --data-root data/ \
  --sequences data/sequences_val.txt \
  --output outputs/submission_full.npz
```

Run **`baseline` with `--export-camera`** (and `--calibration-dir`) when you need refreshed `cameras/<seq>.npz` on disk for the DLT export below (the default `data/cameras/*.npz` from bootstrap is the initial calibration; the tracker updates `K,R,t,k` during `baseline` in memory — export writes the per-frame result).

#### 5. `dlt-export` — `cameras/*.npz` → per-frame `.dlt2d` / `.dlt3d`

Exports vailá-style DLT files (one row per frame) from FIFA camera NPZ files so you can run **`vaila/rec2d.py`** and **`vaila/rec3d.py`** on broadcast / moving-camera pixels.

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--cameras-dir` | Yes | — | Folder containing `*.npz` with keys `K`, `R`, `t`, `k` |
| `--output-dir` | Yes | — | Where to write `<stem>.dlt2d` and/or `<stem>.dlt3d` |
| `--mode` | No | `both` | `2d`, `3d`, or `both` |
| `--undistort-pixels-dir` | No | — | Optional folder of `*.csv`; writes `pixels_undistorted/` under `--output-dir` |

```bash
uv run vaila/vaila_sam.py fifa dlt-export \
  --cameras-dir data/cameras \
  --output-dir outputs/dlt_per_frame \
  --mode both

# Same via module:
uv run python -m vaila.fifa_to_dlt --input data/cameras --output outputs/dlt_per_frame
```

#### 6. `pack` — Split to submission ZIP

| Argument | Required | Description |
|----------|----------|-------------|
| `--submission-full` | Yes | Full submission NPZ from `baseline` |
| `--data-root` | Yes | Data directory (reads `sequences_{split}.txt`) |
| `--output-dir` | Yes | Directory for the output ZIP |
| `--split` | Yes | `val` or `test` |

```bash
# Validation submission
uv run vaila/vaila_sam.py fifa pack \
  --submission-full outputs/submission_full.npz \
  --data-root data/ \
  --output-dir outputs/ \
  --split val

# Test submission
uv run vaila/vaila_sam.py fifa pack \
  --submission-full outputs/submission_full.npz \
  --data-root data/ \
  --output-dir outputs/ \
  --split test
```

### End-to-End Pipeline

```bash
# Step 1: Prepare videos and extract frames
uv run vaila/vaila_sam.py fifa prepare \
  --video-source /path/to/raw_videos --data-root data/

# Step 2: Generate bounding boxes (YOLO or SAM3)
uv run vaila/vaila_sam.py fifa boxes \
  --data-root data/ --sequences data/sequences_val.txt

# Step 3: SAM 3D Body → 2D/3D skeletons (CUDA required)
uv run vaila/vaila_sam.py fifa preprocess \
  --data-root data/ --sequences data/sequences_val.txt

# Step 4: Run baseline → submission NPZ
uv run vaila/vaila_sam.py fifa baseline \
  --data-root data/ --sequences data/sequences_full.txt \
  --output outputs/submission_full.npz

# Step 5a: Pack validation submission
uv run vaila/vaila_sam.py fifa pack \
  --submission-full outputs/submission_full.npz \
  --data-root data/ --output-dir outputs/ --split val

# Step 5b: Pack test submission
uv run vaila/vaila_sam.py fifa pack \
  --submission-full outputs/submission_full.npz \
  --data-root data/ --output-dir outputs/ --split test

# Step 6: Upload outputs/submission_val.zip to Codabench validation portal
#          Upload outputs/submission_test.zip to Codabench test portal
```

### Submitting to Codabench

1. Run the pipeline to generate `submission_val.zip` and/or `submission_test.zip`
2. Go to [Codabench Validation Portal](https://codabench.org/competitions/11681/) (or [Test Portal](https://www.codabench.org/competitions/11682/))
3. Create an account / log in
4. Navigate to **My Submissions**
5. Upload the corresponding ZIP file
6. Wait for evaluation — the primary metric is **MPJPE** (Mean Per Joint Position Error) in millimetres

> **Submission format:** Each ZIP contains a single `submission.npz` with per-sequence arrays of shape `(n_frames, n_persons, 15, 3)` — 15 keypoints with X, Y, Z coordinates in world space.

---

## Main Functions

### vaila_sam.py

- `run_sam3_on_video()` — Core SAM3 video segmentation on a single video
- `run_sam_video()` — GUI entry: Tkinter batch dialog
- `download_sam3_weights_to_vaila_models()` — Download gated SAM3 from HF
- `_resolve_sam3_checkpoint_file()` — Checkpoint auto-detection
- `_maybe_subsample_video_for_vram()` — VRAM auto-sizing
- `_composite_masks_bgr()` — Overlay coloured masks on frames
- `main()` — CLI entry point

### fifa_skeletal_pipeline.py

- `fifa_prepare()` — Copy/convert videos, extract frames
- `fifa_generate_boxes()` — YOLO or SAM3 bounding box generation
- `fifa_preprocess()` — SAM 3D Body to skel_2d / skel_3d
- `fifa_baseline()` — Camera tracker + LBFGS + smoothen to NPZ
- `fifa_pack()` — Split full NPZ to submission ZIP
- `main_fifa_cli()` — CLI dispatcher for FIFA subcommands (includes `dlt-export`)

---

## Full broadcast pipeline (moving camera): SAM → FIFA cameras → DLT → rec2d / rec3d

Use this when the **camera moves** (typical broadcast). You need **one row of DLT coefficients per video frame**, not a single global homography.

| Script | Camera | DLT file shape | When to use |
|--------|--------|----------------|-------------|
| **`vaila/rec2d.py`** | Moving OK | `.dlt2d` with `frame` + 8 coeffs × **N frames** | Broadcast 2D field plane (Z = 0) |
| **`vaila/rec3d.py`** | Moving OK | one `.dlt3d` per camera, each with **N frames** | Multi-view 3D (pixels must be time-aligned across views) |
| `vaila/rec2d_one_dlt2d.py` | **Fixed** only | one row of 8 coeffs | Tripod / static cam |
| `vaila/rec3d_one_dlt3d.py` | **Fixed** only | one row of 11 coeffs per cam | Static multi-cam lab |

**Steps (outline):**

1. **SAM 3 video (batch):** segment all clips and optional point CSVs, e.g.  
   `uv run vaila/vaila_sam.py -i /path/to/Videos/ -o sam_out/ -t player --postprocess-points`  
   (Batch uses **one subprocess per video** by default so CUDA memory resets between clips; see SAM3 skill / `AGENTS.md`.)

2. **FIFA layout:** `fifa bootstrap` / `prepare` so each sequence has `videos/`, `images/`, starter `cameras/<seq>.npz`, `pitch_points.txt`, etc.

3. **Boxes → preprocess → baseline** as in the FIFA section. For DLT export, run baseline **with** `--export-camera` (optionally `--calibration-dir …`) so per-frame `K,R,t,k` are saved to NPZ files you will convert.

4. **DLT export:**  
   `uv run vaila/vaila_sam.py fifa dlt-export --cameras-dir data/cameras --output-dir dlt_out/ --mode both`  
   Optional: `--undistort-pixels-dir path/to/sam_csvs` if radial distortion is non-negligible (see warning printed when `|k|` is large).

5. **2D reconstruction:**  
   `uv run python vaila/rec2d.py --dlt-file dlt_out/<SEQ>.dlt2d --input-dir sam_csv_folder/ --output-dir rec2d_out/ --rate 30`  
   Pixel CSVs must use the header `frame,p1_x,p1_y,...` and frame indices must match rows in the `.dlt2d` file.

6. **3D reconstruction (multi-camera):**  
   `uv run python vaila/rec3d.py --dlt-files dlt_out/camA.dlt3d dlt_out/camB.dlt3d --input-dir merged_pixels/ --output-dir rec3d_out/ --rate 30`  
   You must supply **time-synchronized** pixel CSVs and one **per-frame** `.dlt3d` per camera. The current `rec3d.py` expects one combined CSV directory — if your data are **one CSV per camera**, use `rec3d_one_dlt3d.py` only when cameras are **fixed**; for moving cameras, merge frames into the layout your analysis expects or preprocess columns accordingly.

**GUI:** `FIFA cams→DLT` is available via **Frame B → Soccer Tools → FIFA cams→DLT** (same Tk flow as `vaila.fifa_to_dlt.run_gui_flow()`).

---

## Warnings and Known Issues

- **CUDA required:** Both SAM 3 video and the FIFA pipeline (`preprocess`, `baseline`) require an NVIDIA GPU with CUDA
- **BFloat16 type mismatch:** Some SAM 3 builds produce `Input type (c10::BFloat16) and bias type (float) should be the same` — vailá patches `autocast` and forces FP32 on the tracker backbone, but edge cases may persist
- **Gated repositories:** Both `facebook/sam3` and `facebook/sam-3d-body-dinov3` require Hugging Face license acceptance; 403 errors mean the account is not authorized
- **VRAM OOM:** Long clips load all frames onto GPU; use `--max-frames` or `SAM3_MAX_FRAMES` to cap
- **BPE vocabulary:** SAM 3 PyPI wheel may omit the CLIP BPE file; vailá falls back to `boxmot`'s copy or the package `merges.txt`

---

## Requirements

- Python 3.12
- NVIDIA GPU with CUDA (inference)
- `uv sync --extra sam` — SAM 3 video segmentation
- `uv sync --extra fifa` — FIFA Skeletal Tracking (Lightning, timm, omegaconf)
- `uv sync --extra gpu` — CUDA template for TensorRT/CUDA wheels
- OpenCV (`cv2`), NumPy, Tkinter
- Hugging Face Hub (`huggingface_hub`) for weight downloads

---

## Testing

```bash
# Fast unit tests (no GPU, no weights)
uv run pytest tests/test_vaila_sam.py -v
uv run pytest tests/test_fifa_skeletal_pipeline.py -v
uv run pytest tests/test_fifa_to_dlt.py -v

# GPU smoke test (needs CUDA + sam3 + test1000.mp4)
export VAILA_TEST_SAM_GPU=1
uv run pytest tests/test_vaila_sam.py -v -k sam3_smoke
```

Place a short MP4 at `tests/SAM/test1000.mp4` for smoke tests (see `tests/SAM/README.md`).

---

## Version History

- **v0.3.70 (05 July 2026):** Set H.264/H.265 MP4 formats as the default output videos, keeping AVI only as a legacy fallback option to save disk space.
- **v0.3.69 (05 July 2026):** Help expanded with **Re-ID CLI workflows** (`--stabilize-ids`, `--overlap-frames`, scenario recipes A–H), chunked post-process repair via `sam_postprocess.py`, and AVI overlay / `source=` README support in post-process frame-size detection.
- **v0.3.47 (05 June 2026):** Subprocess-exit diagnostics (SIGKILL/SIGSEGV/SIGABRT/SIGBUS/EXIT_NEEDS_CHUNKING shown by name + actionable hint) in both CLI batch and GUI poller; runtime banner at startup (VRAM, host RAM, effective config, video queue); host-RAM heads-up for long broadcast clips; new `--print-examples` CLI flag + argparse epilog with copy/paste recipes; updated help with **Common errors** matrix.
- **v0.3.43 (07 May 2026):** Fix GUI batch output directory — GUI now passes exact `processed_sam_...` folder to subprocess so no empty sibling folder is created.
- **v0.0.4 (April 2026):** `fifa dlt-export` / `vaila.fifa_to_dlt` — FIFA `cameras/*.npz` → per-frame `.dlt2d` / `.dlt3d` for `rec2d.py` / `rec3d.py` (moving broadcast camera); help section **Full broadcast pipeline**; GUI button **FIFA cams→DLT**
- **v0.0.3 (17 April 2026):** `SamVideoDialog` Help button + editable prompt combobox (14 presets); `SamBatchProgress` Help button; `fifa bootstrap` subcommand (symlinks + sequences + pitch_points); `bin/setup_fifa_sam3d.sh/ps1` for SAM 3D Body cloning and gated weights download; `vaila/fifa_starter_lib/` vendorised (MIT); companion **Soccer-field DLT2D calibration** in `vaila/soccerfield_calib.py`
- **v0.0.2 (April 2026):** FIFA Skeletal Tracking Light pipeline integration; SAM 3.1 Multiplex support; BFloat16 patches; VRAM auto-sizing
- **v0.0.1:** Initial SAM 3 video segmentation with text prompts

### Companion tool: Soccer-field 2D calibration

The `vaila/soccerfield_calib.py` script fits a DLT2D homography from manually
clicked soccer-field keypoints (29 FIFA reference points, 105×68 m). Run:

```bash
uv run vaila/soccerfield_calib.py \
  --video ARG_CRO_000737.mp4 \
  --ref3d models/soccerfield_ref3d.csv \
  --data-root /data/FIFA/data        # drops cameras/<stem>_homography.npz
```

The GUI button **"Soccer-Field Calib"** in vailá's main window (Frame C)
launches the same entrypoint. A Z-vertical extension using goalposts and
anthropometric priors is tracked as a `# TODO: Z vertical (DLT3D)` marker in
the script and will ship in a follow-up.

---

Generated: July 05, 2026
Part of vailá - Multimodal Toolbox
[GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
Contact: paulosantiago@usp.br
