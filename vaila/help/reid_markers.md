# reid_markers

## 📋 Module Information

- **Category:** Processing
- **File:** `vaila/reid_markers.py`
- **Version:** 0.3.108
- **Updated:** 20 August 2026
- **Author:** Adapted from getpixelvideo.py by Prof. Dr. Paulo R. P. Santiago
- **GUI Interface:** ✅ Yes
- **CLI Interface:** ✅ Yes (new, v0.3.102) — `uv run python -u -m vaila.reid_markers --input ...`

## 📖 Description


================================================================================
Marker Re-identification Tool - reid_markers.py
================================================================================
Author: Adapted from getpixelvideo.py by Prof. Dr. Paulo R. P. Santiago
Update Date: 20 August 2026
Version: 0.3.108
Python Version: 3.12.9

Description:
------------
This tool allows correcting identification issues in marker files generated
by getpixelvideo.py. It offers the following functionalities:

1. Marker merging: Combine markers that represent the same object
2. Gap filling: Fill gaps where a marker temporarily disappears
3. Swaps: Fix cases where IDs were swapped in certain frame intervals
4. Geometric ReID: stabilize marker IDs using 2D distance, velocity direction, and optional homography
5. Class-aware YOLO ReID: merge each label independently so balls, rackets,
   and people never compete for the same stable identity slots

================================================================================

### SAM tracking CSV support

If `sam_tracks.csv` is selected, the loader now normalizes SAM long-format tracks to vailá wide marker columns. When a sibling `sam_points.csv` exists, it is used directly; otherwise the loader writes `sam_tracks_reid_points.csv` and `sam_tracks_reid_id_map.csv`.

### Geometric ReID v2: schema auto-detection + `max_ids` merge (v0.3.102)

Upgrades the "Geometric ReID (2D + velocity)" capability to merge an
arbitrary number of fragmented raw tracker ids down into `<= max_ids`
persistent trajectories, and to run **headless** via a new CLI — not just
the swap-fixer GUI flow.

- **Schema auto-detection** (`detect_input_schema`): no manual column
  picking needed for bbox wide-per-slot CSVs (`X_min_person_id_01` etc.,
  the `all_id_detection.csv` convention written by `yolov26track.py`),
  vailá's own `pN_x/pN_y` point convention, row-per-detection bbox
  (`x1,y1,x2,y2` or `x,y,w,h`), and SAM long tracks (`sam_tracks.csv`).
- **`max_ids`**: bounds the number of *concurrently active* identity
  slots (a slot idle longer than `--reid-max-gap` frames frees for reuse
  by a different physical subject entering later — the same mechanism
  serves both "few known subjects" and "peak concurrency, subjects
  enter/exit over time"). Reuses the shared `GeometricFrameLinker`
  (`geometric_reid.py`, the same engine `yolov26track --stabilize-ids`
  and SAM chunk-linking use) via its new `max_tracks` slot pool — no
  second Hungarian/IoU implementation. Omit `--max-ids` to auto-estimate
  from the observed peak simultaneous detections.
- **Class-aware YOLO merge (v0.3.107)**: when a bbox-wide input carries
  `Label_*` columns, ReID runs independently per label and `max_ids` applies
  per class. The output preserves names such as `person_id_01` and
  `sports ball_id_01`; `Tracker ID` is rewritten to the stable ID.
- **No detections are dropped**: every input row keeps a `stable_id`,
  except the one unavoidable case where a single frame genuinely has more
  simultaneous detections than `max_ids` allows — reported honestly via
  `forced_reassignments`/`dropped_rows` in the printed stats rather than
  silently merged.
- **CLI** (headless, no Tk root — safe for scripts/CI):
  ```bash
  uv run python -u -m vaila.reid_markers \
    --input all_id_detection.csv \
    --max-ids 16 \
    --output-dir processed_reid_maxids/
  ```
- **GUI**: the existing "Geometric ReID (2D + velocity)" button now tries
  schema auto-detection first (prompting only for `max_ids`) and falls
  back to the legacy manual-column-selection swap-fixer for files that
  don't match a recognized schema — no workflow regression for existing
  users. Prints the equivalent `>>` CLI command on every run.
- **`yolov26track.py` integration (v0.3.107)**: `--max-ids N` now preserves
  every raw tracklet and automatically writes
  `all_id_detection_reid_maxidsN.csv`. The old persistence-ranked drop cap is
  no longer used because it removed fragments before ReID could match them.
  See `yolov26track.md`.

Validated against a real 16,693-frame tracking CSV: 388 raw fragmented ids
(no live cap) merged to exactly `max_ids=18` (peak concurrency was
actually 21 — `dropped_rows=475` at 18 honestly reported the shortfall;
`max_ids=21` gave `dropped_rows=0`). Separately, on the *same* video's
already-capped-to-16 output (which the live `--stabilize-ids` linker had
grown back to 17 stable ids — an unbounded-slot artifact this `max_tracks`
fix also corrects at the source, see `geometric_reid.md`), this merge
cleanly re-consolidated 17 → 16 with zero dropped rows.


## 🔧 Main Functions

**Total functions found:** 34

- `load_markers_file`
- `is_sam_tracks_file`
- `sam_tracks_to_marker_points`
- `normalize_marker_input`
- `save_markers_file`
- `create_temp_dir`
- `create_temp_file`
- `clear_temp_dir`
- `detect_markers`
- `get_marker_coords`
- `detect_gaps`
- `fill_gaps`
- `merge_markers`
- `swap_markers`
- `save_operations_log`
- `visualize_markers`
- `detect_markers_dynamic`
- `get_marker_coords_dynamic`
- `load_homography_matrix`
- `geometric_reid_align_markers`
- `geometric_reid_align_markers_bidirectional`
- `detect_gaps_dynamic`
- `visualize_markers_dynamic`
- `select_columns_dialog`
- `run_geometric_reid_with_data`
- `create_gui_menu`
- `run_reid_swap_auto_with_data`
- `run_reid_swap_manual_with_data`
- `advanced_reid_gui_with_data`
- `fill_gaps_arima`
- `auto_fill_gaps_arima`




---

📅 **Last Updated:** 20 August 2026 (v0.3.108)
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
