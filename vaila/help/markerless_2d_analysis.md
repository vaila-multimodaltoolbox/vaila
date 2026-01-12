# markerless_2d_analysis.py

## 📋 Module Information

- **Category:** Analysis
- **File:** `vaila/markerless_2d_analysis.py`
- **Lines:** 4259
- **Version:** 0.7.3
- **Author:** Paulo Roberto Pereira Santiago
- **Email:** paulosantiago@usp.br
- **GitHub:** https://github.com/vaila-multimodaltoolbox/vaila
- **GUI Interface:** ✅ Yes
- **License:** AGPLv3.0

## 📖 Description

This script performs batch processing of videos for 2D pose estimation using MediaPipe's Pose model. It processes videos from a specified input directory, overlays pose landmarks on each video frame, and exports both normalized and pixel-based landmark coordinates to CSV files. The script also generates a video with the landmarks overlaid on the original frames.

### Key Features

- **MediaPipe Pose Model**: 33 landmark detection and tracking (MediaPipe Tasks API 0.10.31+)
- **Video Resize Functionality**: Optional upscaling (2x-8x) for better detection
- **Bounding Box (BBox) ROI Selection**: Select rectangular region of interest from first video frame
- **Polygon ROI Selection**: Select free-form polygonal region of interest (irregular shapes)
- **Crop Resize**: Optional upscaling of cropped region for improved detection
- **Initial Frame Padding**: Add repeated frames at start for MediaPipe stabilization
- **Advanced Filtering**: Multiple smoothing algorithms (Butterworth, Savitzky-Golay, LOWESS, Spline, Kalman, ARIMA)
- **Batch Processing**: Process multiple videos in a single operation
- **Memory Management**: Intelligent memory management for large video files
- **CPU Throttling**: Automatic CPU usage monitoring and throttling
- **TOML Configuration**: Save and load configuration files with GUI auto-population
- **Linux Batch Processing**: Automatic batch processing for high-resolution videos
- **Portable Debug Logging**: Cross-platform debug logging system

## 🔧 Main Functions

**Total functions found:** 25+

### Core Processing Functions
- `process_video()` - Main video processing function
- `process_videos_in_directory()` - Batch processing entry point
- `process_video_batch()` - Batch processing for high-res videos
- `should_use_batch_processing()` - Determine if batch processing is needed
- `calculate_batch_size()` - Calculate optimal batch size

### Configuration Functions
- `get_pose_config()` - Get configuration from GUI or TOML
- `get_default_config()` - Get default configuration
- `save_config_to_toml()` - Save configuration to TOML file
- `load_config_from_toml()` - Load configuration from TOML file

### Filtering Functions
- `butter_filter()` - Butterworth filter (low-pass, band-pass)
- `savgol_smooth()` - Savitzky-Golay filter
- `lowess_smooth()` - LOWESS smoothing
- `spline_smooth()` - Spline smoothing
- `kalman_smooth()` - Kalman filter
- `arima_smooth()` - ARIMA model smoothing
- `apply_interpolation_and_smoothing()` - Apply filtering to landmarks
- `apply_temporal_filter()` - Temporal smoothing

### Utility Functions
- `resize_video_opencv()` - Resize video for better detection
- `convert_coordinates_to_original()` - Convert resized coordinates back
- `estimate_occluded_landmarks()` - Estimate occluded landmarks
- `cleanup_memory()` - Memory cleanup for Linux systems
- `get_system_memory_info()` - Get system memory information
- `is_linux_system()` - Check if running on Linux

## 📊 Supported Landmarks (33 points)

### Face (10 landmarks)
- Nose
- Left/Right eye (inner, center, outer)
- Left/Right ear
- Mouth left/right corners

### Upper Body (13 landmarks)
- Left/Right shoulder
- Left/Right elbow
- Left/Right wrist
- Left/Right pinky
- Left/Right index
- Left/Right thumb

### Lower Body (10 landmarks)
- Left/Right hip
- Left/Right knee
- Left/Right ankle
- Left/Right heel
- Left/Right foot index

## ⚙️ Configuration Parameters

### MediaPipe Settings
- **min_detection_confidence** (0.0-1.0): Threshold to start detecting poses
- **min_tracking_confidence** (0.0-1.0): Threshold to keep tracking poses
- **model_complexity** (0-2): 0=fastest, 1=balanced, 2=most accurate
- **enable_segmentation** (True/False): Draw person outline
- **smooth_segmentation** (True/False): Smooth the outline
- **static_image_mode** (True/False): Treat each frame separately
- **apply_filtering** (True/False): Apply built-in smoothing
- **estimate_occluded** (True/False): Guess hidden body parts

### Video Processing Settings
- **enable_resize** (True/False): Upscale video for better detection
- **resize_scale** (2-8): Scale factor (higher = better detection but slower)
- **enable_padding** (True/False): Add initial frames for stabilization
- **pad_start_frames** (0-120): Number of padding frames

### ROI (Region of Interest) Settings
- **enable_crop** (True/False): Enable ROI cropping (BBox or Polygon)
- **BBox ROI**: Rectangular region selection
  - **bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max** (pixels): BBox coordinates
  - Use "Select BBox ROI" button to visually select rectangular region
  - Drag to select, press SPACE/ENTER to confirm, ESC to cancel
- **Polygon ROI**: Free-form polygonal region selection
  - **roi_polygon_points**: List of polygon points [[x1, y1], [x2, y2], ...]
  - Use "Select Polygon ROI" button to visually select polygonal region
  - Left click to add points, right click to remove last point
  - Press ENTER to confirm (minimum 3 points), ESC to cancel, 'R' to reset
- **enable_resize_crop** (True/False): Resize the cropped region for better detection
- **resize_crop_scale** (2-8): Scale factor for cropped region
- **When to use**: Multi-person scenarios, small subjects, excluding background

### Advanced Filtering Settings
- **enable_advanced_filtering** (True/False): Apply smoothing and gap filling
- **interp_method**: `linear`, `cubic`, `nearest`, `kalman`, `none`
- **smooth_method**: `none`, `butterworth`, `savgol`, `lowess`, `kalman`, `splines`, `arima`
- **max_gap** (frames): Maximum gap size to fill

### Smoothing Parameters
- **Butterworth**: `butter_cutoff` (Hz), `butter_fs` (sampling frequency)
- **Savitzky-Golay**: `savgol_window_length`, `savgol_polyorder`
- **LOWESS**: `lowess_frac`, `lowess_it`
- **Kalman**: `kalman_iterations`, `kalman_mode`
- **Spline**: `spline_smoothing_factor`
- **ARIMA**: `arima_p`, `arima_d`, `arima_q`

## 📁 Output Files

For each processed video:

1. **Annotated Video** (`*_mp.mp4`)
   - Original video with pose landmarks overlaid
   - Green circles for landmarks
   - Red lines for connections

2. **Normalized Coordinates** (`*_mp_norm.csv`)
   - 33 landmarks per frame
   - Coordinates normalized to 0-1 scale
   - Format: `frame_index, landmark_x, landmark_y, landmark_z`

3. **Pixel Coordinates** (`*_mp_pixel.csv`)
   - Coordinates in pixel format
   - Original video resolution
   - Format: `frame_index, landmark_x_px, landmark_y_px, landmark_z`

4. **Original Coordinates** (`*_mp_original.csv`)
   - If resize was used, coordinates converted back to original dimensions

5. **vailá Format** (`*_mp_vaila.csv`)
   - Format: `frame, p1_x, p1_y, p2_x, p2_y, ...`

6. **Log File** (`log_info.txt`)
   - Processing metadata
   - Video information
   - Configuration used
   - Detection statistics

7. **Configuration File** (`configuration_used.toml`)
   - All parameters used for processing

## 🚀 Usage

### GUI Mode (Recommended)
```python
from vaila.markerless_2d_analysis import process_videos_in_directory

# Launch GUI for configuration and processing
process_videos_in_directory()
```

### Programmatic Usage
```python
from vaila.markerless_2d_analysis import process_video
from pathlib import Path

config = {
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5,
    'model_complexity': 2,
    'enable_resize': True,
    'resize_scale': 2,
    'enable_advanced_filtering': True,
    'smooth_method': 'butterworth',
    'butter_cutoff': 6.0,
    'butter_fs': 30.0
}

process_video(
    Path('input_video.mp4'),
    Path('output_directory'),
    config
)
```

## 💻 Requirements

### System Requirements
- Python 3.12.12+
- OS: Linux, macOS, Windows
- RAM: 4GB minimum (8GB+ recommended for large videos)

### Python Dependencies
```
opencv-python>=4.8.0
mediapipe>=0.10.0
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
scikit-learn>=1.3.0
statsmodels>=0.14.0
pykalman>=0.9.5
toml>=0.10.2
psutil>=5.9.0
rich>=13.0.0
```

## 🔍 Performance Characteristics

- **Processing Speed**: ~30-60 FPS (CPU, depends on resolution)
- **Memory Usage**: Moderate (batch processing for large videos)
- **Best For**: Single-person, high-quality videos
- **Hardware**: CPU optimized, Linux batch processing

### Batch Processing (Linux)
- Automatically activated for high-resolution videos (>2.7K)
- Batch sizes: 4K+ (20 frames), 2.7K-4K (30 frames), 1080p-2.7K (50 frames)
- Memory cleanup after each batch

## 🐛 Troubleshooting

### Common Issues

1. **Memory Errors**
   - Solution: Use batch processing (automatic on Linux)
   - Alternative: Reduce video resolution or disable resize

2. **Slow Processing**
   - Solution: Use lower model complexity (0 or 1)
   - Alternative: Disable advanced filtering

3. **Poor Detection**
   - Solution: Enable video resize (2x-4x)
   - Alternative: Adjust confidence thresholds (lower = more detections)

4. **Missing Landmarks**
   - Solution: Enable occlusion estimation
   - Alternative: Use advanced filtering to fill gaps

### Performance Tips

- Use resize only when necessary (for small/distant subjects)
- Choose appropriate model complexity for your hardware
- Process videos in smaller batches for better memory management
- Monitor CPU usage and adjust throttling parameters if needed

## 🔗 Integration

### Compatible Modules
- **3D Reconstruction**: Use pixel coordinates with DLT calibration
- **Visualization**: Direct import to vailá plotting modules
- **Data Processing**: Compatible with filtering and interpolation tools
- **Machine Learning**: Training data for pose estimation models

## 📝 Version History

- **v0.7.3** (January 2026): 
  - Added Polygon ROI selection for free-form regions
  - Improved BBox ROI selection with resizable window
  - Added initial frame padding for MediaPipe stabilization
  - Fixed coordinate mapping for both BBox and Polygon ROI
  - Improved ROI coordinate conversion when video resize is enabled
  - Enhanced TOML configuration to save/load ROI settings
- **v0.7.2** (January 2026): 
  - Added bounding box (BBox) ROI selection for small subjects
  - Added crop resize functionality
  - Added initial frame padding option
- **v0.7.1** (January 2026): 
  - Migrated to MediaPipe Tasks API (0.10.31+)
  - Added bounding box (ROI) selection for small subjects
  - Added crop resize functionality
  - Added portable debug logging
  - Improved coordinate mapping for resized videos
  - GUI auto-population from TOML configuration
- **v0.7.0** (November 2025): Added batch processing, improved memory management
- **v0.6.0** (August 2025): Added video resize functionality and advanced filtering
- **v0.5.0**: Added temporal filtering and occlusion estimation
- **v0.4.0**: Added batch processing and memory management
- **v0.3.0**: Added MediaPipe integration with GUI configuration
- **v0.2.0**: Initial implementation with basic pose detection

---

📅 **Generated on:** November 2025  
🔗 **Part of vailá - Multimodal Toolbox**  
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
