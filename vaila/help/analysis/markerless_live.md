# markerless_live

## 📋 Module Information

- **Category:** Analysis
- **File:** `vaila/markerless_live.py`
- **Lines:** 1486
- **Size:** 56506 characters
- **Version:** 0.0.2
- **Author:** Moser José (https://moserjose.com/),  Prof. Dr. Paulo Santiago
- **GUI Interface:** ✅ Yes

## 📖 Description


Script: markerless_live.py
Author: Moser José (https://moserjose.com/),  Prof. Dr. Paulo Santiago
Version: 0.0.2
Created: April 9, 2025
Last Updated: April 11, 2025

Description:
This script performs real-time pose estimation and angle calculation using either YOLO or
MediaPipe's Pose model. It provides a graphical interface for selecting the pose detection
engine and configuring its parameters, allowing users to analyze movement in real-time
through their webcam.

Features:
- Dual engine support:
    - YOLO (better for multiple people detection)
    - MediaPipe (faster, optimized for single person)
- Real-time angle calculations for various body joints
- Configurable parameters for each engine:
    - YOLO: confidence threshold, model selection
    - MediaPipe: model complexity, detection confidence
- Visual feedback:
    - Skeleton overlay
    - Joint angles display
    - Person detection bounding box
- Data export:
    - CSV files with angle measurements
    - Angle plots over time
...

## 🔧 Main Functions

**Total functions found:** 12

- `list_available_cameras`
- `download_model`
- `run_markerless_live`
- `calculate_angle`
- `adapt_to_keypoint_structure`
- `process_keypoints`
- `process_frame`
- `get_model_path`
- `detect_yolo_model_type`
- `process_frame`
- `save_data`
- `run`




---

📅 **Generated automatically on:** 08/10/2025 14:24:24
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
