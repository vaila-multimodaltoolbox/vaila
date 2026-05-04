# Markerless 2D YOLOv26 Pose Help

This tool provides 2D pose estimation using YOLOv26/v11 models.

## Quick Start
1. **Model**: Select a YOLO pose model. `yolo11x-pose.pt` offers the highest accuracy.
2. **Confidence**: Set the minimum detection threshold (default 0.5).
3. **Directory**: Choose the folder containing your videos.

## Outputs
Results are saved in a folder named `processed_yolo26_YYYYMMDD_HHMMSS`:
- **Video**: Visual verification of tracking quality.
- **Normalized CSV**: Use this for cross-video comparisons.
- **Pixel CSV**: Use this for spatial measurements in pixels.

## 17 Keypoints Diagram
The model tracks 17 joints following the COCO standard.

- **Head**: Nose, Eyes, Ears (0-4)
- **Upper Body**: Shoulders, Elbows, Wrists (5-10)
- **Lower Body**: Hips, Knees, Ankles (11-16)

---
© 2026 vailá Multimodal Toolbox
