# Markerless 3D Analysis

## Overview

The Markerless 3D Analysis module provides advanced 3D pose estimation from monocular video using a sophisticated pipeline that combines 2D pose detection, 3D lifting, ground plane anchoring, and optional DLT3D refinement.

## Pipeline Overview

The module implements a complete 3D reconstruction pipeline:

1. **2D Pose Detection**: Extract 2D landmarks using MediaPipe
2. **3D Lifting**: Use VideoPose3D to lift 2D poses to 3D
3. **Ground Anchoring**: Anchor poses to ground plane using DLT2D calibration
4. **Scale Calibration**: Calibrate vertical scale using participant leg length
5. **Optional DLT3D Refinement**: Further refine 3D coordinates using multi-camera calibration

## Key Features

- **Monocular 3D Reconstruction**: Generate 3D pose data from single camera videos
- **Batch Processing**: Process multiple videos or CSV files simultaneously
- **Multiple Input Formats**: Support for video files and pre-extracted 2D CSV data
- **Flexible Calibration**: Support for DLT2D ground plane calibration and DLT3D multi-camera calibration
- **TOML Configuration**: Comprehensive configuration management with TOML files
- **GUI Interface**: User-friendly graphical interface for parameter configuration
- **C3D Export**: Export results in C3D format compatible with motion capture software

## Supported Pose Models

- **COCO17 Format**: 17 keypoints (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles)
- **MediaPipe Mapping**: Automatic conversion from MediaPipe's 33 landmarks to COCO17 format

## Configuration Parameters

### Input/Output Settings
- **Input Directory**: Directory containing video files or CSV data
- **Output Directory**: Directory for processed results
- **File Patterns**: Support for common video formats and CSV files
- **Units**: Metric (meters) or imperial (inches) units

### Calibration Settings
- **DLT2D Path**: Path to 2D camera calibration file for ground plane anchoring
- **DLT3D Path**: Optional path to 3D multi-camera calibration file
- **Leg Length**: Participant leg length for vertical scale calibration (default: 0.42m)

### Processing Options
- **VideoPose3D Model**: Choose from different pretrained models
- **Batch Size**: Processing batch size for VideoPose3D inference
- **Temporal Smoothing**: Apply temporal filtering to 3D coordinates
- **Ground Anchoring**: Enable/disable ground plane anchoring
- **DLT3D Refinement**: Enable/disable multi-camera refinement

## Output Files

For each processed input, the module generates:

1. **3D Coordinates CSV** (`*_3d.csv`): 3D pose coordinates in meters
2. **C3D File** (`*_3d.c3d`): C3D format file compatible with motion capture software
3. **Processing Log** (`*_log.txt`): Detailed processing information and statistics
4. **Configuration File** (`config.toml`): Complete configuration used for processing

## Usage

### GUI Mode (Recommended)

```python
from vaila.markerless_3d_analysis import process_videos_in_directory

# Launch GUI for configuration and processing
process_videos_in_directory()
```

### Programmatic Usage

```python
from vaila.markerless_3d_analysis import run_single_video

# Define configuration
config = {
    'input_dir': '/path/to/videos',
    'output_dir': '/path/to/output',
    'dlt2d_path': '/path/to/calibration.dlt2d',
    'dlt3d_path': '/path/to/calibration3d.dlt3d',
    'leg_length_m': 0.42,
    'units': 'm',
    'batch_size': 16,
    'temporal_smoothing': True
}

# Process single video
run_single_video(config, 'input_video.mp4', '/path/to/output')
```

### Batch Processing

```python
import glob
from vaila.markerless_3d_analysis import run_single_video

# Process all videos in directory
video_files = glob.glob('/path/to/videos/*.mp4')
for video_path in video_files:
    run_single_video(config, video_path, '/path/to/output')
```

## Calibration Setup

### DLT2D Ground Plane Calibration

1. **Capture Calibration Video**: Record a video of a known calibration pattern
2. **Extract 2D Points**: Use the DLT2D calibration tool to extract 2D coordinates
3. **Generate DLT2D File**: Create the calibration file using the DLT2D module

### DLT3D Multi-Camera Calibration (Optional)

1. **Multi-Camera Setup**: Set up multiple synchronized cameras
2. **Calibration Pattern**: Use a known 3D calibration object
3. **Extract 2D/3D Points**: Extract corresponding 2D and 3D coordinates
4. **Generate DLT3D File**: Create the calibration file using the DLT3D module

## Requirements

### Core Dependencies
- Python 3.11+
- NumPy (`numpy`)
- OpenCV (`opencv-python`)
- MediaPipe (`mediapipe`)
- SciPy (`scipy`)

### 3D Lifting (VideoPose3D)
- PyTorch (`torch`)
- VideoPose3D model files (automatically downloaded)

### C3D Export
- ezc3d (`ezc3d`)

### Configuration
- TOML support (`tomli` for Python < 3.11, built-in for Python 3.11+)

## Performance Considerations

### Processing Speed
- **2D Extraction**: ~30-60 FPS depending on video resolution
- **3D Lifting**: ~10-20 FPS with GPU acceleration
- **Calibration**: ~1-5 seconds per video

### Memory Usage
- **Batch Processing**: Adjust batch size based on available RAM
- **Large Videos**: Consider processing in segments for very long videos

### Hardware Recommendations
- **GPU**: Recommended for VideoPose3D inference (significant speedup)
- **RAM**: 8GB+ recommended for batch processing
- **Storage**: ~10x input video size for output files

## Accuracy Considerations

### Sources of Error
- **Camera Calibration**: Inaccurate calibration affects absolute accuracy
- **Leg Length Estimation**: Incorrect leg length affects vertical scale
- **Occlusion**: Partial occlusions reduce tracking accuracy
- **Lighting**: Poor lighting conditions affect 2D detection

### Improving Accuracy
- Use high-quality camera calibration
- Ensure proper leg length measurement
- Optimize lighting conditions
- Use multiple camera views when possible

## Integration with vailá Ecosystem

This module integrates with other vailá tools:

- **Motion Capture**: Use with cluster analysis or full-body mocap tools
- **Visualization**: Compatible with 3D plotting and C3D viewing tools
- **Data Processing**: Output can be processed with filtering and smoothing tools
- **Machine Learning**: 3D coordinates can be used for ML model training

## Troubleshooting

### Common Issues

1. **VideoPose3D Model Download**: Ensure internet connection for automatic model download
2. **Calibration File Errors**: Verify calibration file format and parameters
3. **Memory Errors**: Reduce batch size or process videos individually
4. **GPU Issues**: Ensure CUDA compatibility or use CPU-only mode

### Performance Optimization

- Use GPU acceleration when available
- Process videos in smaller batches
- Disable unnecessary processing steps for quick results
- Use pre-extracted 2D CSV data when available

## Version History

- **v0.6.0**: Added DLT3D refinement and improved calibration options
- **v0.5.0**: Added VideoPose3D integration and ground anchoring
- **v0.4.0**: Added batch processing and TOML configuration
- **v0.3.0**: Added GUI interface and CSV input support
- **v0.2.0**: Initial implementation with basic 3D lifting
- **v0.1.0**: Proof of concept with simple monocular reconstruction
