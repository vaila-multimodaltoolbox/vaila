# Markerless 2D YOLOv26 Analysis

This module performs standalone 2D markerless analysis using **YOLOv26 (YOLOv11-Pose)** models. It is designed to be a lightweight alternative to the MediaPipe-based analysis, focusing on the 17 standard COCO keypoints.

## Features

- **Standard 17 Keypoints**: Detects nose, eyes, ears, shoulders, elbows, wrists, hips, knees, and ankles.
- **YOLOv26/v11 Support**: Compatible with all YOLOv11-Pose and YOLOv26-Pose models (nano to extra-large).
- **GPU Acceleration**: Automatically detects and uses NVIDIA CUDA or Apple Silicon (MPS) for faster processing.
- **Batch Processing**: Processes all videos in a selected directory automatically.
- **Output Files**:
  - `*_yolo26.mp4`: Annotated video with skeleton overlay.
  - `*_yolo26_norm.csv`: Normalized coordinates (0.0 to 1.0).
  - `*_yolo26_pixel.csv`: Pixel coordinates in video resolution.
  - `log_info.txt`: Metadata and processing summary.

## Usage

1. Launch the module from the **Markerless 2D** button in the main `vailá` interface.
2. Select **YOLOv26 Pose Only (17 keypoints)**.
3. Configure the YOLO model and confidence threshold.
4. Select the directory containing your videos.
5. The script will process each video and save results in a timestamped subdirectory.

## Keypoints (COCO Format)

| Index | Name | Index | Name |
|-------|------|-------|------|
| 0 | nose | 9 | left_wrist |
| 1 | left_eye | 10 | right_wrist |
| 2 | right_eye | 11 | left_hip |
| 3 | left_ear | 12 | right_hip |
| 4 | right_ear | 13 | left_knee |
| 5 | left_shoulder | 14 | right_knee |
| 6 | right_shoulder | 15 | left_ankle |
| 7 | left_elbow | 16 | right_ankle |
| 8 | right_elbow | | |

## Troubleshooting

- **No Detections**: Lower the confidence threshold if the subject is far away or lighting is poor.
- **Performance**: Use an NVIDIA GPU for the best processing speeds.
- **Multiple People**: This module currently exports data for the first person detected in each frame. For multi-person analysis, use the **Advanced** markerless module.
