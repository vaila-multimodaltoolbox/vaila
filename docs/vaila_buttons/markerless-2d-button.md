# Markerless 2D Analysis - Button B1_r1_c4

## Overview

The **Markerless 2D Analysis** button (B1_r1_c4) in the vailá GUI provides access to advanced 2D pose estimation capabilities using state-of-the-art computer vision models. This module offers two processing modes: **Standard** (MediaPipe only) and **Advanced** (YOLOv11 + MediaPipe), allowing users to choose the optimal balance between speed and accuracy for their specific use case.

## Button Location

- **Grid Position**: B1_r1_c4 (Button 1, Row 1, Column 4)
- **GUI Category**: Markerless Analysis
- **Access Path**: Main GUI → Markerless 2D Analysis

## Available Versions

### Version 1: Standard (MediaPipe Only - CPU)
- **Script**: `vaila/markerless_2d_analysis.py`
- **Speed**: Faster processing (CPU optimized)
- **Use Case**: Single-person scenarios, real-time applications
- **Accuracy**: High for single-person detection
- **Features**:
  - MediaPipe Pose model (33 landmarks)
  - Video resize functionality (2x-8x upscaling)
  - Advanced filtering (Butterworth, Savitzky-Golay, LOWESS, Spline, Kalman, ARIMA)
  - Batch processing with memory management
  - CPU throttling for resource optimization
  - TOML configuration support
  - Bounding box (ROI) selection for small subjects
  - Portable debug logging

### Version 1 GPU: Standard (MediaPipe Only - NVIDIA GPU)
- **Script**: `vaila/markerless_2d_analysis_nvidia.py`
- **Speed**: Much faster processing (GPU accelerated)
- **Use Case**: Single-person scenarios, high-performance requirements
- **Accuracy**: High for single-person detection (same as CPU version)
- **Features**:
  - All features from Version 1 (CPU)
  - **NVIDIA GPU acceleration** via MediaPipe GPU delegate
  - **Automatic GPU detection** and testing
  - **Device selection dialog** (CPU/GPU choice at startup)
  - GPU information display (name, driver, memory)
  - Automatic fallback to CPU if GPU unavailable
  - **Requirements**: NVIDIA GPU with CUDA support and drivers

### Version 2: Advanced (YOLOv11 + MediaPipe)
- **Script**: `vaila/markerless2d_analysis_v2.py`
- **Speed**: Slower but more robust
- **Use Case**: Multi-person scenarios, complex environments
- **Accuracy**: Superior for multi-person and occluded scenarios
- **Features**:
  - YOLOv11 person detection + MediaPipe pose estimation
  - YOLO11-pose models (nano, small, medium, large, extra-large)
  - YOLO-only mode (17 keypoints from YOLO11-pose)
  - YOLO+MediaPipe hybrid mode
  - GPU/CPU automatic detection
  - Temporal filtering (Kalman, Savitzky-Golay)
  - Enhanced multi-person tracking

## Technical Specifications

### Supported Input Formats
- **Video Formats**: `.mp4`, `.avi`, `.mov`
- **Resolution**: Any (automatic batch processing for high-res videos)
- **Frame Rate**: Any (automatically detected)

### Output Files

For each processed video, the module generates:

1. **Annotated Video** (`*_mp.mp4`)
   - Original video with pose landmarks overlaid
   - Green circles for landmarks
   - Red lines for connections
   - Optional bounding boxes (YOLO mode)

2. **Normalized Coordinates** (`*_mp_norm.csv`)
   - 33 landmarks (MediaPipe) or 17 keypoints (YOLO-only)
   - Coordinates normalized to 0-1 scale
   - Format: `frame_index, landmark_x, landmark_y, landmark_z`

3. **Pixel Coordinates** (`*_mp_pixel.csv`)
   - Coordinates in pixel format
   - Original video resolution
   - Format: `frame_index, landmark_x_px, landmark_y_px, landmark_z`

4. **Log File** (`log_info.txt`)
   - Processing metadata
   - Hardware configuration
   - Pipeline configuration (MediaPipe/YOLO)
   - Detection statistics
   - Performance metrics

5. **Configuration File** (`configuration_used.toml`)
   - All parameters used for processing
   - Reusable for batch processing

## Landmark Detection

### MediaPipe Landmarks (33 points)
- **Face**: Nose, eyes (inner, center, outer), ears, mouth corners
- **Upper Body**: Shoulders, elbows, wrists, hands (pinky, index, thumb)
- **Lower Body**: Hips, knees, ankles, heels, feet

### YOLO11-Pose Keypoints (17 points)
- **Face**: Nose, left/right eye, left/right ear
- **Upper Body**: Left/right shoulder, elbow, wrist
- **Lower Body**: Left/right hip, knee, ankle

## Configuration Parameters

### MediaPipe Settings
- **min_detection_confidence** (0.0-1.0): Threshold to start detecting poses
- **min_tracking_confidence** (0.0-1.0): Threshold to keep tracking poses
- **model_complexity** (0-2): 0=fastest, 1=balanced, 2=most accurate
- **enable_segmentation** (True/False): Draw person outline
- **smooth_segmentation** (True/False): Smooth the outline
- **static_image_mode** (True/False): Treat each frame separately
- **apply_filtering** (True/False): Apply built-in smoothing
- **estimate_occluded** (True/False): Guess hidden body parts

### YOLO Settings (Version 2 only)
- **use_yolo** (True/False): Enable YOLO person detection
- **yolo_mode**: 
  - `yolo_only`: Use only YOLO11-pose (17 keypoints)
  - `yolo_mediapipe`: Use YOLO for detection + MediaPipe for pose (33 landmarks)
- **yolo_model**: Model selection
  - `yolo11n-pose.pt`: Nano (fastest, smallest)
  - `yolo11s-pose.pt`: Small
  - `yolo11m-pose.pt`: Medium
  - `yolo11l-pose.pt`: Large
  - `yolo11x-pose.pt`: Extra Large (most accurate)
- **yolo_conf** (0.0-1.0): YOLO confidence threshold

### Video Processing Settings
- **enable_resize** (True/False): Upscale video for better detection
- **resize_scale** (2-8): Scale factor (higher = better detection but slower)
- **enable_padding** (True/False): Add initial frames for stabilization
- **pad_start_frames** (0-120): Number of padding frames

### Advanced Filtering (Version 1 only)
- **enable_advanced_filtering** (True/False): Apply smoothing and gap filling
- **interp_method**: `linear`, `cubic`, `nearest`, `kalman`, `none`
- **smooth_method**: `none`, `butterworth`, `savgol`, `lowess`, `kalman`, `splines`, `arima`
- **max_gap** (frames): Maximum gap size to fill

### Temporal Filtering (Version 2 only)
- **filter_type**: `none`, `kalman`, `savgol`

## Performance Characteristics

### Version 1 (Standard - CPU)
- **Processing Speed**: ~30-60 FPS (CPU, depends on resolution)
- **Memory Usage**: Moderate (batch processing for large videos)
- **Best For**: Single-person, high-quality videos
- **Hardware**: CPU optimized, Linux batch processing

### Version 1 GPU (Standard - NVIDIA GPU)
- **Processing Speed**: ~60-150+ FPS (GPU, depends on GPU model and resolution)
- **Memory Usage**: Moderate (GPU memory optimized)
- **Best For**: Single-person, high-performance requirements, batch processing
- **Hardware**: NVIDIA GPU with CUDA support required
- **Speedup**: 2-5x faster than CPU version (depending on GPU)

### Version 2 (Advanced)
- **Processing Speed**: ~10-30 FPS (CPU), ~60-120 FPS (GPU)
- **Memory Usage**: Higher (YOLO model loading)
- **Best For**: Multi-person, complex scenarios, occlusions
- **Hardware**: GPU recommended, CPU fallback available

## Usage Workflow

### Step 1: Launch Module
1. Click **Markerless 2D Analysis** button (B1_r1_c4)
2. Select version:
   - **1**: Standard (MediaPipe only - CPU)
   - **1 GPU**: Standard (MediaPipe only - NVIDIA GPU) - **NEW!**
   - **2**: Advanced (YOLOv11 + MediaPipe)

### Step 2: Device Selection (Version 1 GPU only)
1. **Automatic GPU Detection**: The script automatically detects NVIDIA GPU availability
2. **GPU Testing**: MediaPipe GPU delegate is tested automatically
3. **Device Selection Dialog**: Choose between:
   - **CPU**: Standard processing (always available)
   - **GPU**: NVIDIA CUDA acceleration (if available and tested)
4. **GPU Information Display**: Shows GPU name, driver version, memory, and test status
5. **Automatic Fallback**: If GPU test fails, CPU is used automatically

### Step 3: Configure Parameters
1. Select input directory containing videos
2. Select output base directory
3. Configure detection parameters via GUI or load TOML file

### Step 4: Process Videos
1. Module processes all videos in input directory
2. Progress displayed in terminal
3. GPU/CPU usage information shown (Version 1 GPU)
4. Output files saved to timestamped directory

### Step 5: Review Results
1. Check annotated videos for quality
2. Review CSV files for coordinate data
3. Examine log files for statistics

## Integration with vailá Ecosystem

### Data Flow
```
Video Input → Pose Detection → Coordinate Extraction → CSV Export
                ↓
         Annotated Video
                ↓
    Integration with other modules:
    - DLT Calibration (3D reconstruction)
    - Visualization (2D/3D plotting)
    - Machine Learning (model training)
    - Multimodal Analysis (IMU, force plate, etc.)
```

### Compatible Modules
- **3D Reconstruction**: Use pixel coordinates with DLT calibration
- **Visualization**: Direct import to vailá plotting modules
- **Data Processing**: Compatible with filtering and interpolation tools
- **Machine Learning**: Training data for pose estimation models

## Requirements

### System Requirements
- **Python**: 3.12.12+
- **OS**: Linux, macOS, Windows
- **RAM**: 4GB minimum (8GB+ recommended for large videos)
- **GPU**: 
  - **Version 1 GPU**: NVIDIA GPU with CUDA support required
  - **Version 2**: Optional but recommended (NVIDIA GPU with CUDA)
  - **Version 1 (CPU)**: No GPU required

### GPU Requirements (Version 1 GPU)
- **NVIDIA GPU**: Any CUDA-capable NVIDIA GPU
- **NVIDIA Drivers**: Latest drivers installed
- **CUDA Toolkit**: Required for MediaPipe GPU delegate
- **MediaPipe**: Version 0.10.31+ with GPU delegate support
- **Testing**: Automatic GPU detection and MediaPipe delegate testing

### Python Dependencies
```bash
# Core dependencies
opencv-python>=4.8.0
mediapipe>=0.10.0
numpy>=1.24.0
pandas>=2.0.0

# Version 1 additional
scipy>=1.10.0
scikit-learn>=1.3.0
statsmodels>=0.14.0
pykalman>=0.9.5
toml>=0.10.2
psutil>=5.9.0
rich>=13.0.0

# Version 2 additional
ultralytics>=8.0.0
torch>=2.0.0
```

## Troubleshooting

### Common Issues

1. **Memory Errors**
   - **Solution**: Use batch processing (automatic on Linux)
   - **Alternative**: Reduce video resolution or disable resize

2. **Slow Processing**
   - **Solution**: Use lower model complexity or smaller YOLO model
   - **Alternative**: 
     - Use Version 1 GPU (NVIDIA GPU) for 2-5x speedup
     - Enable GPU acceleration (Version 2)

3. **Poor Detection**
   - **Solution**: Enable video resize (2x-4x)
   - **Alternative**: Adjust confidence thresholds
   - **For multi-person**: Use Version 2 (Advanced)

4. **Missing Landmarks**
   - **Solution**: Enable occlusion estimation
   - **Alternative**: Use advanced filtering to fill gaps

### Performance Optimization Tips

- **Single-person videos**: 
  - Use Version 1 GPU (NVIDIA GPU) for best performance
  - Use Version 1 (CPU) if no GPU available
- **Multi-person videos**: Use Version 2 (Advanced)
- **High-resolution videos**: Enable batch processing (automatic)
- **Low-quality videos**: Enable resize (2x-4x)
- **Real-time applications**: Use Version 1 GPU with model_complexity=0
- **GPU Acceleration**: 
  - Version 1 GPU provides 2-5x speedup over CPU
  - Automatic GPU detection and testing ensures compatibility
  - Fallback to CPU if GPU unavailable or test fails

## Version History

### Version 0.7.1 (Current - Standard CPU)
- Added batch processing for Linux
- Improved memory management
- Enhanced filtering options
- TOML configuration support
- Bounding box (ROI) selection for small subjects
- Portable debug logging

### Version 0.7.1 (Current - Standard GPU) - **NEW!**
- **NVIDIA GPU acceleration** via MediaPipe GPU delegate
- **Automatic GPU detection** using nvidia-smi
- **MediaPipe GPU delegate testing** before use
- **Device selection dialog** for CPU/GPU choice
- **GPU information display** (name, driver, memory)
- **Automatic fallback** to CPU if GPU unavailable
- All features from CPU version
- **2-5x performance improvement** over CPU version

### Version 0.0.2 (Current - Advanced)
- YOLO11-pose integration
- Multi-person detection
- GPU/CPU automatic detection
- YOLO-only mode support
- Enhanced temporal filtering

## References

- **MediaPipe**: [Google MediaPipe Pose](https://google.github.io/mediapipe/solutions/pose)
- **YOLOv11**: [Ultralytics YOLOv11](https://docs.ultralytics.com/models/yolo11/)
- **vailá Repository**: [GitHub](https://github.com/vaila-multimodaltoolbox/vaila)

## Support

For issues, questions, or contributions:
- **Email**: paulosantiago@usp.br
- **GitHub Issues**: [vaila-multimodaltoolbox/vaila/issues](https://github.com/vaila-multimodaltoolbox/vaila/issues)
- **Documentation**: [vailá Documentation](https://vaila-multimodaltoolbox.github.io/vaila/)

---

**Last Updated**: November 2025  
**Maintained by**: Paulo Roberto Pereira Santiago  
**License**: AGPLv3.0

