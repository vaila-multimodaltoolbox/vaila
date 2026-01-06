# markerless_2d_analysis_nvidia.py

## 📋 Module Information

- **Category:** Utils
- **File:** `vaila/markerless_2d_analysis_nvidia.py`
- **Lines:** 3902
- **Version:** 0.7.1 (NVIDIA GPU Optimized)
- **Author:** Paulo Roberto Pereira Santiago
- **Email:** paulosantiago@usp.br
- **GitHub:** https://github.com/vaila-multimodaltoolbox/vaila
- **GUI Interface:** ✅ Yes
- **License:** AGPLv3.0

## 📖 Description

This script performs batch processing of videos for 2D pose estimation using MediaPipe's Pose model with **NVIDIA GPU acceleration**. It is identical to the CPU version (`markerless_2d_analysis.py`) but optimized for NVIDIA GPUs using MediaPipe GPU delegate, providing **2-5x performance improvement** over CPU processing.

### Key Features

- **NVIDIA GPU Acceleration**: MediaPipe GPU delegate for CUDA-enabled GPUs
- **Automatic GPU Detection**: Detects NVIDIA GPU availability using nvidia-smi
- **MediaPipe GPU Delegate Testing**: Automatically tests GPU compatibility before use
- **Device Selection Dialog**: Choose between CPU or GPU at startup
- **GPU Information Display**: Shows GPU name, driver version, memory, and test status
- **Automatic Fallback**: Falls back to CPU if GPU unavailable or test fails
- **All CPU Version Features**: Includes all features from the CPU version:
  - MediaPipe Pose model (33 landmarks) - Tasks API 0.10.31+
  - Video resize functionality (2x-8x upscaling)
  - Bounding box (ROI) selection for small subjects
  - Crop resize functionality
  - Advanced filtering (Butterworth, Savitzky-Golay, LOWESS, Spline, Kalman, ARIMA)
  - Batch processing with memory management
  - CPU throttling for resource optimization
  - TOML configuration support with GUI auto-population
  - Portable debug logging

## 🔧 Main Functions

**Total functions found:** 30+

### GPU Detection Functions
- `detect_nvidia_gpu()` - Detect NVIDIA GPU availability and get GPU information
- `test_mediapipe_gpu()` - Test MediaPipe GPU delegate compatibility

### Device Selection
- `DeviceSelectionDialog` - GUI dialog for CPU/GPU selection

### Core Processing Functions
- `process_video()` - Main video processing function (with GPU support)
- `process_videos_in_directory()` - Batch processing entry point
- `process_video_batch()` - Batch processing for high-res videos
- `process_frame_with_tasks_api()` - Frame processing with Tasks API
- `should_use_batch_processing()` - Determine if batch processing is needed
- `calculate_batch_size()` - Calculate optimal batch size

### Configuration Functions
- `get_pose_config()` - Get configuration from GUI or TOML
- `get_default_config()` - Get default configuration
- `save_config_to_toml()` - Save configuration to TOML file
- `load_config_from_toml()` - Load configuration from TOML file
- `populate_fields_from_config()` - Populate GUI from loaded TOML

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
- `map_landmarks_to_full_frame()` - Map landmarks from crop to full frame
- `estimate_occluded_landmarks()` - Estimate occluded landmarks
- `get_model_path()` - Download MediaPipe Tasks models
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

### Bounding Box (ROI) Settings
- **enable_crop** (True/False): Enable bounding box cropping
- **bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max** (pixels): ROI coordinates
- **enable_resize_crop** (True/False): Resize the cropped region
- **resize_crop_scale** (2-8): Scale factor for cropped region
- **Visual ROI Selection**: Use "Select ROI from Video" button in GUI to visually select region

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
   - Blue lines for connections
   - Yellow rectangle for bounding box (if enabled)

2. **Normalized Coordinates** (`*_mp_norm.csv`)
   - 33 landmarks per frame
   - Coordinates normalized to 0-1 scale
   - Format: `frame_index, landmark_x, landmark_y, landmark_z`

3. **Pixel Coordinates** (`*_mp_pixel.csv`)
   - Coordinates in pixel format
   - Original video resolution
   - Format: `frame_index, landmark_x_px, landmark_y_px, landmark_z`

4. **vailá Format** (`*_mp_vaila.csv`)
   - Format: `frame, p1_x, p1_y, p2_x, p2_y, ...`

5. **Log File** (`log_info.txt`)
   - Processing metadata
   - Video information
   - Configuration used
   - Detection statistics
   - GPU/CPU usage information

6. **Configuration File** (`configuration_used.toml`)
   - All parameters used for processing

## 🚀 Usage

### GUI Mode (Recommended)

1. **Launch Script**: Run the script to open the GUI
2. **Device Selection**: 
   - Script automatically detects NVIDIA GPU
   - Tests MediaPipe GPU delegate compatibility
   - Shows device selection dialog:
     - **CPU**: Standard processing (always available)
     - **GPU**: NVIDIA CUDA acceleration (if available and tested)
   - Displays GPU information (name, driver, memory, test status)
3. **Configure Parameters**: Select input/output directories and configure parameters
4. **Process Videos**: Script processes all videos with selected device

```python
from vaila.markerless_2d_analysis_nvidia import process_videos_in_directory

# Launch GUI for configuration and processing
# Device selection dialog will appear automatically
process_videos_in_directory()
```

### Programmatic Usage

```python
from vaila.markerless_2d_analysis_nvidia import process_video
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

# use_gpu=True for GPU, use_gpu=False for CPU
process_video(
    Path('input_video.mp4'),
    Path('output_directory'),
    config,
    use_gpu=True  # Use GPU if available
)
```

## 💻 Requirements

### System Requirements
- **Python**: 3.12.12+
- **OS**: Linux, macOS, Windows
- **RAM**: 4GB minimum (8GB+ recommended for large videos)
- **GPU**: **NVIDIA GPU with CUDA support required** (for GPU mode)

### GPU Requirements
- **NVIDIA GPU**: Any CUDA-capable NVIDIA GPU
- **NVIDIA Drivers**: Latest drivers installed
- **CUDA Toolkit**: Required for MediaPipe GPU delegate
- **MediaPipe**: Version 0.10.31+ with GPU delegate support
- **Testing**: Automatic GPU detection and MediaPipe delegate testing

### Python Dependencies
```
opencv-python>=4.8.0
mediapipe>=0.10.31  # Must support GPU delegate
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

### GPU Mode (Recommended)
- **Processing Speed**: ~60-150+ FPS (GPU, depends on GPU model and resolution)
- **Memory Usage**: Moderate (GPU memory optimized)
- **Best For**: Single-person, high-performance requirements, batch processing
- **Hardware**: NVIDIA GPU with CUDA support required
- **Speedup**: **2-5x faster than CPU version** (depending on GPU)

### CPU Mode (Fallback)
- **Processing Speed**: ~30-60 FPS (CPU, depends on resolution)
- **Memory Usage**: Moderate (batch processing for large videos)
- **Best For**: Systems without NVIDIA GPU
- **Hardware**: CPU optimized, Linux batch processing

### Batch Processing (Linux)
- Automatically activated for high-resolution videos (>2.7K)
- Batch sizes: 4K+ (20 frames), 2.7K-4K (30 frames), 1080p-2.7K (50 frames)
- Memory cleanup after each batch

## 🐛 Troubleshooting

### GPU Issues

1. **GPU Not Detected**
   - **Solution**: Ensure NVIDIA drivers are installed (`nvidia-smi` should work)
   - **Check**: Run `nvidia-smi` in terminal to verify GPU detection

2. **MediaPipe GPU Delegate Test Fails**
   - **Solution**: Script automatically falls back to CPU
   - **Check**: Ensure MediaPipe version 0.10.31+ is installed
   - **Check**: Verify CUDA Toolkit is properly installed

3. **GPU Performance Not Better**
   - **Solution**: Check GPU utilization with `nvidia-smi` during processing
   - **Check**: Ensure GPU has sufficient memory
   - **Alternative**: Use CPU mode if GPU is slower (rare)

### Common Issues

1. **Memory Errors**
   - **Solution**: Use batch processing (automatic on Linux)
   - **Alternative**: Reduce video resolution or disable resize

2. **Slow Processing**
   - **Solution**: Use GPU mode (if available) for 2-5x speedup
   - **Alternative**: Use lower model complexity (0 or 1)
   - **Alternative**: Disable advanced filtering

3. **Poor Detection**
   - **Solution**: Enable video resize (2x-4x)
   - **Solution**: Use bounding box (ROI) selection for small subjects
   - **Alternative**: Adjust confidence thresholds (lower = more detections)

4. **Missing Landmarks**
   - **Solution**: Enable occlusion estimation
   - **Alternative**: Use advanced filtering to fill gaps

### Performance Tips

- **Use GPU mode** for best performance (2-5x speedup)
- Use resize only when necessary (for small/distant subjects)
- Use bounding box (ROI) selection for multi-person scenarios
- Choose appropriate model complexity for your hardware
- Process videos in smaller batches for better memory management
- Monitor GPU/CPU usage and adjust parameters if needed

## 🔗 Integration

### Compatible Modules
- **3D Reconstruction**: Use pixel coordinates with DLT calibration
- **Visualization**: Direct import to vailá plotting modules
- **Data Processing**: Compatible with filtering and interpolation tools
- **Machine Learning**: Training data for pose estimation models

### Data Flow
```
Video Input → GPU/CPU Processing → Pose Detection → Coordinate Extraction → CSV Export
                ↓
         Annotated Video
                ↓
    Integration with other modules:
    - DLT Calibration (3D reconstruction)
    - Visualization (2D/3D plotting)
    - Machine Learning (model training)
    - Multimodal Analysis (IMU, force plate, etc.)
```

## 📝 Version History

- **v0.7.1** (January 2026): 
  - **NVIDIA GPU acceleration** via MediaPipe GPU delegate
  - **Automatic GPU detection** and testing
  - **Device selection dialog** for CPU/GPU choice
  - **GPU information display** (name, driver, memory)
  - **Automatic fallback** to CPU if GPU unavailable
  - Migrated to MediaPipe Tasks API (0.10.31+)
  - Added bounding box (ROI) selection
  - Added crop resize functionality
  - Added portable debug logging
  - Improved coordinate mapping
  - GUI auto-population from TOML
- **v0.7.0** (November 2025): Added batch processing, improved memory management
- **v0.6.0** (August 2025): Added video resize functionality and advanced filtering
- **v0.3.0** (April 2025): Initial NVIDIA GPU version

## 🔗 Related Scripts

- **CPU Version**: `markerless_2d_analysis.py` - Standard CPU processing
- **Advanced Version**: `markerless2d_analysis_v2.py` - YOLO + MediaPipe

---

📅 **Last Updated:** January 2026  
🔗 **Part of vailá - Multimodal Toolbox**  
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
