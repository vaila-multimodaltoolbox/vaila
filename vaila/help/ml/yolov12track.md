# yolov12track

## 📋 Module Information

- **Category:** Ml
- **File:** `vaila/yolov12track.py`
- **Lines:** 1383
- **Size:** 50161 characters
- **Version:** 0.0.3
- **Author:** Paulo Roberto Pereira Santiago
- **GUI Interface:** ✅ Yes

## 📖 Description


Project: vailá
Script: yolov12track.py

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 18 February 2025
Update Date: 02 Augst 2025
Version: 0.0.3

Description:
    This script performs object detection and tracking on video files using the YOLO model v12.
    It integrates multiple features, including:
      - Object detection and tracking using the Ultralytics YOLO library.
      - A graphical interface (Tkinter) for dynamic parameter configuration.
      - Video processing with OpenCV, including drawing bounding boxes and overlaying tracking data.
      - Generation of CSV files containing frame-by-frame tracking information per tracker ID.
      - Video conversion to more compatible formats using FFmpeg.

Usage:
    Run the script from the command line by passing the path to a video file as an argument:
            python yolov12track.py

Requirements:
    - Python 3.x
    - OpenCV
    - PyT...

## 🔧 Main Functions

**Total functions found:** 20

- `initialize_csv`
- `update_csv`
- `get_hardware_info`
- `detect_optimal_device`
- `validate_device_choice`
- `standardize_filename`
- `process_video`
- `get_color_for_id_improved`
- `get_color_palette`
- `get_color_for_id`
- `create_combined_person_csv`
- `run_yolov12track`
- `body`
- `show_help`
- `hide_help`
- `validate`
- `apply`
- `body`
- `browse_custom_model`
- `on_tab_changed`




---

📅 **Generated automatically on:** 08/10/2025 14:24:24
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
