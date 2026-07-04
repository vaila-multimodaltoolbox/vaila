# reid_yolotrack

## Module Information

- **Category:** Utils / Re-ID
- **File:** `vaila/reid_yolotrack.py`
- **Version:** 0.3.68
- **Updated:** 04 July 2026
- **Author:** Paulo Roberto Pereira Santiago
- **GUI Interface:** Yes (Tkinter directory + threshold dialog)
- **CLI Interface:** Yes (headless via `run_appearance_reid_on_tracking_dir`)

## Description

Post-track **appearance Re-ID** using OSNet embeddings (via `boxmot`). Merges fragmented IDs that geometric Re-ID cannot resolve (same uniform, long re-entry gaps).

### When to use

| Scenario | Recommended layer |
|----------|------------------|
| Short flicker / crossing | Geometric linker (`yolov26track --stabilize-ids`) |
| Same uniform / re-entry after many seconds | **This module** (`--appearance-reid`) |
| Broadcast with known pitch | Geometric + homography gate |

### Integration with yolov26track (v0.3.68)

The module is now callable as an **optional post-pass** after geometric stabilize:

```bash
uv run python -m vaila.yolov26track track \
  --source video.mp4 --appearance-reid --appearance-reid-threshold 0.6
```

Or from the GUI: check **"Post-track OSNet appearance ReID"** in Run Mode.

### CSV filename compatibility (v0.3.68)

Parser now handles both:
- **`person_id_01.csv`** (yolov26track v0.3.x output)
- **`person_id3.csv`** (legacy format)

Aggregates (`all_id_detection.csv`, `*_pose.csv`, `yolo_reid_links.csv`) are auto-skipped.

## Main Functions

- `parse_tracking_csv_filename(filename)` — Parse per-ID CSV names (both formats)
- `iter_tracking_csv_paths(directory)` — Discover valid per-ID CSVs
- `_get_reid_model(weights, device)` — Lazy boxmot OSNet loader
- `ReidProcessor` — Full pipeline: load CSVs → extract features → cluster → write corrected
- `run_appearance_reid_on_tracking_dir(tracking_dir, video_path, threshold)` — **Headless entry point** for yolov26track integration
- `run_reid_yolotrack()` — GUI entry point (Tkinter)

## Requirements

- `boxmot` (optional extra; not in core `uv sync` — install with `uv add boxmot`)
- `torch`, `opencv-python`, `pandas`, `rich`

## Outputs

- `reid_corrected/` subdirectory with merged per-ID CSVs (`{label}_id_{NN:02d}.csv`)
- `reid_corrected/reid_corrected_video.mp4` — visualization with corrected IDs

---

📅 **Last Updated:** 04 July 2026 (v0.3.68)
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
