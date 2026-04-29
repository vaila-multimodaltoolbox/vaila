# Field KPs (AI) Button

The **Field KPs (AI)** button launches the `soccerfield_keypoints_ai.py` module, which provides an automatic first pass of soccer-field keypoints in pixel space.

## Overview

This tool uses deep learning (YOLO-pose or Roboflow Inference) to detect 32 canonical soccer pitch keypoints in broadcast video frames or images. These keypoints serve as a seed for:
- Manual refinement in **Get Pixel Coord**
- Homography fitting in **Soccer-Field Calib**
- Dataset expansion for retraining pitch detection models

## Key Features

- **Dual Backends:** 
  - **Ultralytics:** Uses a local `.pt` pose model (bundled weights: `pitch32_recipeA_400ep/best.pt`).
  - **Roboflow:** Connects to the Roboflow Inference API (requires an API key).
- **Two Modes:**
  - **Frame Mode:** Analyzes a single frame from a video.
  - **Video Mode:** Batch processes a video with a configurable stride (e.g., every 10th frame).
- **Output Formats:**
  - `field_keypoints_video.csv`: Detailed confidence and coordinate data.
  - `field_keypoints_getpixelvideo.csv`: Wide-format CSV compatible with the **Get Pixel Coord** tool.
  - `field_keypoints_overlay.mp4`: Visualization video with detected markers.

## Usage

1. Click **Field KPs (AI)** in Frame B.
2. Select the input video and output folder.
3. Choose the mode (**frame** or **video**).
4. Configure the confidence threshold (default 0.3) and image size (1280 recommended).
5. Click **Run**.
6. The resulting `field_keypoints_getpixelvideo.csv` can be loaded directly into **Get Pixel Coord** for manual correction or into **Soccer-Field Calib** for homography fitting.

## Troubleshooting

- **No markers detected:** Lower the confidence threshold or ensure the video has clear pitch lines.
- **CUDA OOM:** Reduce the `imgsz` or use a smaller YOLO model.
- **Backend Error:** Ensure `ultralytics` or `inference` packages are installed (`uv sync`).

---
See also: [FIFA Workflow](../../docs/fifa_workflow.md)
