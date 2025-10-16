# API Reference - Modules

## Overview

This page provides a comprehensive reference of all modules available in the vailá Multimodal Toolbox. Each module is organized by category and includes brief descriptions and usage information.

## Multimodal Analysis Modules

### Markerless Analysis
- **`markerless_2d_analysis`**: 2D pose estimation using MediaPipe with advanced filtering and video resize capabilities
- **`markerless_3d_analysis`**: 3D pose reconstruction from monocular video using VideoPose3D
- **`markerless_live`**: Real-time markerless pose tracking and analysis

### Motion Capture
- **`cluster_analysis`**: Analysis of anatomical marker clusters for trunk and pelvis kinematics
- **`mocap_analysis`**: Full-body motion capture data processing and analysis

### Sensor Analysis
- **`imu_analysis`**: Inertial Measurement Unit data processing and orientation estimation
- **`forceplate_analysis`**: Ground reaction force and center of pressure analysis
- **`emg_labiocom`**: Electromyography signal processing and analysis
- **`gnss_analysis`**: GPS/GNSS trajectory analysis and mapping

## Data Processing Tools

### File Management
- **`filemanager`**: Comprehensive file operations (rename, copy, move, transfer)
- **`rearrange_data`**: CSV data reorganization and column manipulation
- **`readc3d_export`**: C3D file reading and CSV export functionality
- **`readcsv_export`**: CSV to C3D conversion tools

### Data Filtering and Processing
- **`interp_smooth_split`**: Data interpolation, smoothing, and segmentation
- **`filtering`**: Advanced signal filtering algorithms
- **`filter_utils`**: Butterworth and other filter implementations
- **`gapfill_split`**: Gap filling and data splitting utilities

### 3D Reconstruction
- **`dlt2d`**: 2D Direct Linear Transformation camera calibration
- **`dlt3d`**: 3D Direct Linear Transformation multi-camera calibration
- **`rec2d`**: 2D reconstruction from calibrated cameras
- **`rec3d`**: 3D reconstruction from multiple calibrated cameras

### Marker Processing
- **`reid_markers`**: Marker re-identification and tracking
- **`modifylabref`**: Laboratory reference system modifications

## Video Processing Tools

### Video Manipulation
- **`videoprocessor`**: Video merging, splitting, and batch processing
- **`extractpng`**: Video frame extraction to PNG images
- **`cutvideo`**: Video cutting and trimming operations
- **`resize_video`**: Video resolution modification tools

### Video Analysis
- **`getpixelvideo`**: Pixel coordinate extraction from videos
- **`numberframes`**: Video frame counting and metadata extraction
- **`syncvid`**: Multi-camera video synchronization
- **`drawboxe`**: Region of interest definition in videos

### Video Compression
- **`compress_videos_h264`**: H.264 video compression
- **`compress_videos_h265`**: H.265/HEVC video compression
- **`compress_videos_h266`**: H.266/VVC video compression

## Visualization Tools

### 2D Plotting
- **`vailaplot2d`**: Comprehensive 2D plotting with multiple chart types
- **`soccerfield`**: Soccer field visualization tools

### 3D Visualization
- **`vailaplot3d`**: 3D data plotting and visualization
- **`viewc3d`**: C3D file visualization tools
- **`showc3d`**: Interactive C3D data viewer

## Machine Learning Tools

### YOLO Integration
- **`yolov11track`**: YOLOv11-based object tracking
- **`yolov12track`**: YOLOv12-based object tracking
- **`yolotrain`**: YOLO model training interface

### ML Walkway
- **`vaila_mlwalkway`**: Machine learning gait analysis pipeline
- **`walkway_ml_prediction`**: ML model prediction for gait parameters

### Specialized ML
- **`ml_models_training`**: General ML model training utilities
- **`ml_valid_models`**: ML model validation tools

## Specialized Analysis Tools

### Biomechanical Analysis
- **`vaila_and_jump`**: Vertical jump analysis and performance metrics
- **`animal_open_field`**: Animal behavior analysis in open field tests
- **`cube2d_kinematics`**: 2D kinematic analysis of cube movements
- **`vector_coding`**: Vector coding analysis for joint coupling

### Medical Imaging
- **`usound_biomec1`**: Ultrasound image analysis for biomechanical applications
- **`brainstorm`**: Brain signal analysis tools
- **`scout_vaila`**: Scout analysis integration

### Specialized Processing
- **`markerless2d_mpyolo`**: MediaPipe + YOLO integration for pose estimation
- **`mphands`**: Hand pose estimation and analysis
- **`mpangles`**: Angle calculation from pose landmarks
- **`stabilogram_analysis`**: Postural stability analysis

## Utility Modules

### Core Utilities
- **`vaila_manifest`**: Core vailá functionality and manifest
- **`common_utils`**: Common utility functions
- **`utils`**: General utility functions
- **`data_processing`**: Data processing utilities

### Dialogs and UI
- **`dialogsuser`**: User interface dialog components
- **`dialogsuser_cluster`**: Cluster-specific dialog components
- **`native_file_dialog`**: Native file dialog integration

### External Integrations
- **`vaila_ytdown`**: YouTube video downloading tools
- **`vaila_iaudiovid`**: Audio-video insertion tools
- **`rm_duplicateframes`**: Duplicate frame removal
- **`vaila_upscaler`**: Video upscaling tools

## Module Categories Summary

| Category | Modules | Description |
|----------|---------|-------------|
| **Markerless Analysis** | 3 | 2D/3D pose estimation from video |
| **Motion Capture** | 2 | Traditional motion capture processing |
| **Sensor Analysis** | 4 | IMU, force plate, EMG, GNSS data |
| **Data Processing** | 15+ | File management, filtering, calibration |
| **Video Processing** | 10+ | Video manipulation and analysis |
| **Visualization** | 4 | 2D/3D plotting and data display |
| **Machine Learning** | 8+ | YOLO, ML walkway, model training |
| **Specialized Tools** | 12+ | Domain-specific analysis tools |
| **Utilities** | 10+ | Core utilities and integrations |

## Module Dependencies

### Core Dependencies (Most Modules)
- `numpy`: Numerical computations
- `pandas`: Data manipulation
- `matplotlib`: Plotting and visualization
- `scipy`: Scientific computing

### Specialized Dependencies
- **Computer Vision**: `opencv-python`, `mediapipe`
- **3D Graphics**: `pyvista`, `open3d`
- **Machine Learning**: `torch`, `tensorflow`, `scikit-learn`
- **Video Processing**: `ffmpeg-python`
- **Motion Capture**: `ezc3d`

## Getting Started with Modules

### Basic Usage Pattern

```python
# Import module
from vaila import module_name

# Use main function (usually GUI-based)
module_name.main_function()

# Or access specific functions
from vaila.module_name import specific_function
specific_function(parameters)
```

### GUI vs Programmatic Usage

Most modules offer both:
- **GUI Mode**: Interactive interface for parameter selection
- **Programmatic Mode**: Direct function calls for automation

### Configuration Files

Many modules use:
- **TOML Files**: For complex configuration (markerless_3d_analysis)
- **JSON Files**: For parameter storage (calibration modules)
- **CSV Files**: For data input/output

## Module Development

### Adding New Modules

1. Create module file in `vaila/` directory
2. Follow naming conventions (`vaila_*` for main modules)
3. Include comprehensive docstrings
4. Add GUI interface when applicable
5. Include errorrrr handling and logging

### Module Integration

- Use `vaila.py` as integration point
- Follow established patterns for GUI integration
- Include proper errorrrr handling and user feedback

## Troubleshooting

### Common Module Issues

1. **Import Errors**: Check dependency installation
2. **GUI Issues**: Verify Tkinter installation and display settings
3. **Memory Issues**: Use appropriate batch sizes for large datasets
4. **File Path Issues**: Use absolute paths for reliability

### Performance Optimization

- **Batch Processing**: Process data in appropriate chunk sizes
- **Memory Management**: Clear large variables when not needed
- **Hardware Acceleration**: Use GPU when available (YOLO, compression)
- **Caching**: Utilize built-in caching for repeated operations

## Version Information

All modules follow semantic versioning with the main vailá release. Module-specific changes are tracked in individual module docstrings and version histories.

For detailed information about specific modules, refer to the individual module documentation pages in the navigation menu.