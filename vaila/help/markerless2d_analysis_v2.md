# markerless2d_analysis_v2.py

## 📋 Module Information

- **Category:** Machine Learning / Analysis
- **File:** `vaila/markerless2d_analysis_v2.py`
- **Lines:** 1396
- **Version:** 0.0.2
- **Author:** Paulo Roberto Pereira Santiago
- **Email:** paulosantiago@usp.br
- **GitHub:** https://github.com/vaila-multimodaltoolbox/vaila
- **GUI Interface:** ✅ Yes
- **License:** AGPLv3.0

## 📖 Description

Advanced version of the 2D pose estimation script that incorporates YOLOv11 for person detection to improve MediaPipe's performance, especially for multi-person scenarios. This version offers two processing modes: **YOLO-only** (using YOLO11-pose models) and **YOLO+MediaPipe** (hybrid approach).

### Key Features

- **YOLOv11 Integration**: Person detection using YOLOv11 models
- **YOLO11-Pose Models**: Support for nano, small, medium, large, and extra-large pose models
- **Dual Processing Modes**:
  - `yolo_only`: Use only YOLO11-pose (17 keypoints)
  - `yolo_mediapipe`: Use YOLO for detection + MediaPipe for pose (33 landmarks)
- **GPU/CPU Support**: Automatic hardware detection and optimization
- **Temporal Filtering**: Kalman and Savitzky-Golay filters
- **Model Selection**: Choose from 5 YOLO11-pose model sizes
- **Enhanced Multi-Person Tracking**: Better handling of multiple people in frame

## 🔧 Main Functions

**Total functions found:** 15+

### Core Processing Functions
- `process_video()` - Main video processing function
- `process_videos_in_directory()` - Batch processing entry point
- `process_frame_with_yolo_pose_only()` - Process frame using YOLO11-pose only
- `process_frame_with_mediapipe()` - Process frame with MediaPipe (with optional YOLO)
- `detect_persons_with_yolo()` - Detect persons using YOLO

### YOLO Functions
- `download_or_load_yolo_model()` - Download or load YOLO model
- `download_yolo_model()` - Download YOLO model from repository
- `get_hardware_info()` - Get detailed hardware information

### Filtering Functions
- `apply_temporal_filter()` - Apply temporal filter to landmarks
- `apply_kalman_filter()` - Kalman filter implementation
- `apply_savgol_filter()` - Savitzky-Golay filter implementation

### Visualization Functions
- `draw_yolo_landmarks()` - Custom drawing for YOLO landmarks

### Configuration Functions
- `get_pose_config()` - Get configuration from GUI

## 📊 Supported Landmarks

### MediaPipe Mode (33 landmarks)
- Same as `markerless_2d_analysis.py`

### YOLO-Only Mode (17 keypoints)
- **Face**: Nose, left/right eye, left/right ear
- **Upper Body**: Left/right shoulder, elbow, wrist
- **Lower Body**: Left/right hip, knee, ankle

## ⚙️ Configuration Parameters

### MediaPipe Settings
- **min_detection_confidence** (0.0-1.0): Threshold to start detecting poses
- **min_tracking_confidence** (0.0-1.0): Threshold to keep tracking poses
- **model_complexity** (0-2): 0=fastest, 1=balanced, 2=most accurate
- **enable_segmentation** (True/False): Draw person outline
- **smooth_segmentation** (True/False): Smooth the outline
- **static_image_mode** (True/False): Treat each frame separately

### YOLO Settings
- **use_yolo** (True/False): Enable YOLO person detection
- **yolo_mode**: 
  - `yolo_only`: Use only YOLO11-pose (17 keypoints)
  - `yolo_mediapipe`: Use YOLO for detection + MediaPipe for pose (33 landmarks)
- **yolo_model**: Model selection
  - `yolo11n-pose.pt`: Nano (fastest, ~6MB)
  - `yolo11s-pose.pt`: Small (~19MB)
  - `yolo11m-pose.pt`: Medium (~52MB)
  - `yolo11l-pose.pt`: Large (~104MB)
  - `yolo11x-pose.pt`: Extra Large (most accurate, ~209MB)
- **yolo_conf** (0.0-1.0): YOLO confidence threshold

### Temporal Filtering
- **filter_type**: 
  - `none`: No filtering
  - `kalman`: Kalman filter (good for tracking)
  - `savgol`: Savitzky-Golay filter (good for smoothing)
  - `median`: Moving Median filter (best for removing outliers/spikes)

### Graphical User Interface (GUI)
The configuration dialog includes the following buttons:
- **Save Config**: Save current settings to a `.toml` file.
- **Load Config**: Load settings from a `.toml` file.
- **Create Config**: Reset settings to defaults.
- **Help**: Open this help documentation.

## 📁 Output Files

For each processed video:

1. **Annotated Video** (`*_mp.mp4`)
   - Original video with pose landmarks overlaid
   - Green circles for landmarks
   - Red lines for connections
   - Optional bounding boxes (YOLO+MediaPipe mode)

2. **Normalized Coordinates** (`*_mp_norm.csv`)
   - 33 landmarks (MediaPipe mode) or 17 keypoints (YOLO-only)
   - Coordinates normalized to 0-1 scale

3. **Pixel Coordinates** (`*_mp_pixel.csv`)
   - Coordinates in pixel format
   - Original video resolution

    - Original video resolution
 
 4. **YOLO Keypoints CSV** (`*_yolo_pixel.csv`)
    - Raw 17 keypoints from YOLO model (if used)
    - Pixel coordinates and confidence
 
 5. **Configuration File** (`config.toml`)
    - Copy of the configuration parameters used
 
 6. **Report File** (`report.txt`)
    - Hardware information
    - Video information
    - Pipeline configuration (YOLO/MediaPipe)
    - Detection statistics
    - Performance metrics

## 🚀 Usage

### GUI Mode (Recommended)
```python
from vaila.markerless2d_analysis_v2 import process_videos_in_directory

# Launch GUI for configuration and processing
process_videos_in_directory()
```

### Programmatic Usage
```python
from vaila.markerless2d_analysis_v2 import process_video, download_or_load_yolo_model
from pathlib import Path

# Load YOLO model
yolo_model = download_or_load_yolo_model("yolo11x-pose.pt")

config = {
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5,
    'model_complexity': 2,
    'use_yolo': True,
    'yolo_mode': 'yolo_mediapipe',
    'yolo_model': 'yolo11x-pose.pt',
    'yolo_conf': 0.5,
    'filter_type': 'kalman'
}

process_video(
    Path('input_video.mp4'),
    Path('output_directory'),
    config,
    yolo_model
)
```

## 💻 Requirements

### System Requirements
- Python 3.12.12+
- OS: Linux, macOS, Windows
- RAM: 4GB minimum (8GB+ recommended)
- GPU: Optional but recommended (CUDA-capable)

### Python Dependencies
```
opencv-python>=4.8.0
mediapipe>=0.10.0
numpy>=1.24.0
pandas>=2.0.0
ultralytics>=8.0.0
torch>=2.0.0
scipy>=1.10.0
pykalman>=0.9.5
```

## 🔍 Performance Characteristics

- **Processing Speed**: 
  - CPU: ~10-30 FPS
  - GPU: ~60-120 FPS
- **Memory Usage**: Higher (YOLO model loading)
- **Best For**: Multi-person, complex scenarios, occlusions
- **Hardware**: GPU recommended, CPU fallback available

### Model Performance Comparison

| Model | Size | Speed (CPU) | Speed (GPU) | Accuracy |
|-------|------|-------------|-------------|----------|
| yolo11n-pose | ~6MB | Fastest | Fastest | Good |
| yolo11s-pose | ~19MB | Fast | Fast | Better |
| yolo11m-pose | ~52MB | Medium | Medium | Good |
| yolo11l-pose | ~104MB | Slow | Fast | Very Good |
| yolo11x-pose | ~209MB | Slowest | Fast | Best |

## 🐛 Troubleshooting

### Common Issues

1. **YOLO Model Not Loading**
   - Solution: Check internet connection for first-time download
   - Alternative: Manually download model to `vaila/models/` directory

2. **GPU Not Detected**
   - Solution: Install CUDA-enabled PyTorch
   - Alternative: Script automatically falls back to CPU

3. **Memory Errors**
   - Solution: Use smaller YOLO model (nano or small)
   - Alternative: Process videos individually

4. **Poor Multi-Person Detection**
   - Solution: Use `yolo_mediapipe` mode
   - Alternative: Adjust YOLO confidence threshold

### Performance Tips

- **Single-person videos**: Use `yolo_only` mode with nano model
- **Multi-person videos**: Use `yolo_mediapipe` mode with larger model
- **GPU available**: Use extra-large model for best accuracy
- **CPU only**: Use nano or small model for acceptable speed

## 🔗 Integration

### Compatible Modules
- **3D Reconstruction**: Use pixel coordinates with DLT calibration
- **Visualization**: Direct import to vailá plotting modules
- **Machine Learning**: Training data for pose estimation models
- **Multi-Person Tracking**: Enhanced tracking with YOLO detection

## 📝 Version History

- **v0.0.2** (November 2025): 
  - Added YOLO-only mode
  - Added YOLO11-pose model selection
  - Enhanced terminal feedback
  - Fixed NaN handling
  - Custom drawing for YOLO landmarks
- **v0.0.1**: Initial release with YOLO+MediaPipe integration

## 🔬 Technical Details

### YOLO11-Pose Keypoint Mapping

YOLO11-pose provides 17 keypoints that are mapped to MediaPipe's 33-landmark format:

- YOLO keypoint 0 (nose) → MediaPipe landmark 0
- YOLO keypoint 1 (left_eye) → MediaPipe landmark 2
- YOLO keypoint 2 (right_eye) → MediaPipe landmark 5
- ... (see code for complete mapping)

### Processing Pipeline

1. **YOLO Detection**: Detect persons in frame
2. **Pose Estimation**: 
   - YOLO-only: Extract 17 keypoints from YOLO11-pose
   - YOLO+MediaPipe: Use YOLO bbox to guide MediaPipe pose estimation
3. **Temporal Filtering**: Apply Kalman or Savitzky-Golay filter
4. **Coordinate Conversion**: Convert to normalized and pixel formats
5. **Visualization**: Draw landmarks on frame

---

📅 **Generated on:** November 2025  
🔗 **Part of vailá - Multimodal Toolbox**  
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
