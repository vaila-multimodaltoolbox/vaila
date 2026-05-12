# usound_biomec1

## 📋 Module Information

- **Category:** Tools
- **File:** `vaila\usound_biomec1.py`
- **Lines:** 1570
- **Size:** 53500 characters
- **Version:** 0.3.44

- **GUI Interface:** ✅ Yes

## 📖 Description


usound_biomec1.py

Module to analyze ultrasound images with manual thickness measurements and
smarter before/after batch comparison support.

Updated by: Prof. Paulo R. P. Santiago
Updated: 12 May 2026
Version: 0.3.44

- Separate BEFORE and AFTER directory selection for comparison workflows
- Parent-folder batch mode for muscle/before and muscle/after structures
- Automatic ROI detection with manual fallback for ultrasound content
- Condition-aware measurement CSV and summary reports
- Smarter visual comparisons based on best cross-condition image matches


## 🔧 Main Functions

**Total functions found:** 39

- `list_images`
- `sanitize_label`
- `load_image_or_raise`
- `infer_comparison_label`
- `discover_before_after_groups`
- `write_batch_summary`
- `build_condition_records`
- `detect_ultrasound_roi`
- `combine_rois`
- `compute_shared_roi`
- `preview_roi`
- `select_roi_manually`
- `choose_roi_and_scale`
- `crop_image_records`
- `redraw_annotations`
- `mouse_event`
- `write_measurement_row`
- `process_images`
- `crop_images_batch`
- `adjust_edge_parameters`
- `create_comparison_images`
- `_fit_label`
- `side_by_side_images`
- `overlay_images`
- `overlay_with_edges`
- `edges_only_comparison`
- `process_condition_image_records`
- `_measurement_stats`
- `write_condition_summaries`
- `prepare_similarity_image`
- `calculate_image_similarity`
- `build_comparison_plan`
- `select_unique_best_pairs`
- `save_comparison_plan`
- `create_before_after_comparison_images`
- `run_before_after_workflow`
- `run_parent_batch_workflow`
- `choose_usound_workflow_mode`
- `run_usound`
- `on_crop`
- `on_threshold1`
- `on_threshold2`
- `on_blur`
- `update_display`




---

📅 **Generated automatically on:** 12/05/2026 16:15:00
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
