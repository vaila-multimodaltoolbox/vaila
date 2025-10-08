# reid_yolotrack

## 📋 Module Information

- **Category:** Utils
- **File:** `vaila/reid_yolotrack.py`
- **Lines:** 607
- **Size:** 21894 characters
- **Version:** 0.1.0
- **Author:** Paulo Roberto Pereira Santiago
- **GUI Interface:** ✅ Yes

## 📖 Description


Project: vailá Multimodal Toolbox
Script: reid_yolotrack.py - ReID with YOLOTrack

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 29 July 2024
Update Date: 10 August 2025
Version: 0.1.0

Description:
This script performs batch processing of videos for ReID using YOLOTrack. It processes videos from a specified input directory,
and exports both normalized and pixel-based landmark coordinates to CSV files. The script also generates a
video with the landmarks overlaid on the original frames.

Usage:
    python -m vaila.reid_yolotrack
    or
    python -m vaila.reid_yolotrack --input_dir path/to/input_directory
    or
    python -m vaila.reid_yolotrack --input_dir path/to/input_directory --output_dir path/to/output_directory
    or
    python -m vaila.reid_yolotrack --input_dir path/to/input_directory --output_dir path/to/output_directory --reid_threshold 0.6
    or
    python -m vaila.reid_yolotrac...

## 🔧 Main Functions

**Total functions found:** 12

- `get_color_for_id`
- `get_rgb_color_string`
- `run_reid_yolotrack`
- `get_reid_model`
- `load_tracking_data`
- `extract_reid_features`
- `compute_average_features`
- `cluster_similar_ids`
- `create_corrected_csvs`
- `create_visualization_video`
- `process`
- `get_reid_model`




---

📅 **Generated automatically on:** 08/10/2025 14:24:24
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
