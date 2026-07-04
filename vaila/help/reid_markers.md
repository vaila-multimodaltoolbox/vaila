# reid_markers

## 📋 Module Information

- **Category:** Processing
- **File:** `vaila\reid_markers.py`
- **Lines:** 2585
- **Size:** 94499 characters
- **Version:** 0.3.68
- **Author:** Adapted from getpixelvideo.py by Prof. Dr. Paulo R. P. Santiago
- **GUI Interface:** ✅ Yes

## 📖 Description


================================================================================
Marker Re-identification Tool - reid_markers.py
================================================================================
Author: Adapted from getpixelvideo.py by Prof. Dr. Paulo R. P. Santiago
Update Date: 09 June 2026
Version: 0.3.47
Python Version: 3.12.9

Description:
------------
This tool allows correcting identification issues in marker files generated
by getpixelvideo.py. It offers the following functionalities:

1. Marker merging: Combine markers that represent the same object
2. Gap filling: Fill gaps where a marker temporarily disappears
3. Swaps: Fix cases where IDs were swapped in certain frame intervals
4. Geometric ReID: stabilize marker IDs using 2D distance, velocity direction, and optional homography

================================================================================

### SAM tracking CSV support

If `sam_tracks.csv` is selected, the loader now normalizes SAM long-format tracks to vailá wide marker columns. When a sibling `sam_points.csv` exists, it is used directly; otherwise the loader writes `sam_tracks_reid_points.csv` and `sam_tracks_reid_id_map.csv`.


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

📅 **Last Updated:** 04 July 2026 (v0.3.68)
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
