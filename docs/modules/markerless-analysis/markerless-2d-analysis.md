# Markerless 2D Analysis

## Overview

The Markerless 2D Analysis module provides advanced 2D pose estimation using MediaPipe's Pose model. This module processes videos to detect and track human pose landmarks, generating both normalized and pixel-based coordinate data along with annotated videos.

## Features

- **Batch Video Processing**: Process multiple videos in a single operation
- **MediaPipe Integration**: Uses Google's MediaPipe Pose model for accurate pose detection
- **Video Resize Functionality**: Optional upscaling (2x-8x) for better detection in low-resolution videos
- **Advanced Filtering**: Multiple smoothing algorithms including Butterworth, Savitzky-Golay, LOWESS, Spline, Kalman, and ARIMA filters
- **Occlusion Estimation**: Automatic estimation of occluded landmarks based on anatomical constraints
- **Coordinate Conversion**: Automatic conversion back to original video dimensions after resize processing
- **Memory Management**: Intelligent memory management to handle large video files
- **CPU Throttling**: Automatic CPU usage monitoring and throttling for resource management

## Supported Landmarks

The module detects and tracks 33 pose landmarks:

- **Face**: Nose, eyes (inner, center, outer), ears, mouth corners
- **Torso**: Shoulders, elbows, wrists, hands (pinky, index, thumb)
- **Lower Body**: Hips, knees, ankles, heels, feet

## Configuration Parameters

### MediaPipe Settings
- **Detection Confidence**: Minimum confidence threshold for pose detection (default: 0.5)
- **Tracking Confidence**: Minimum confidence threshold for pose tracking (default: 0.5)
- **Model Complexity**: MediaPipe model complexity level (0: Light, 1: Full, 2: Heavy)
- **Segmentation**: Enable pose segmentation (default: False)
- **Smooth Segmentation**: Enable smooth segmentation output (default: True)

### Processing Settings
- **Video Resize Factor**: Upscaling factor for better detection (1x to 8x)
- **Temporal Filtering**: Enable temporal smoothing of landmarks
- **Occlusion Estimation**: Enable estimation of occluded landmarks
- **Smoothing Algorithm**: Choose from multiple smoothing algorithms
- **Filter Parameters**: Configurable filter parameters based on selected algorithm

## Output Files

For each processed video, the module generates:

1. **Annotated Video** (`*_mp.mp4`): Original video with pose landmarks overlaid
2. **Normalized Coordinates** (`*_mp_norm.csv`): Landmark coordinates normalized (0-1 scale)
3. **Pixel Coordinates** (`*_mp_pixel.csv`): Landmark coordinates in pixel format
4. **Original Coordinates** (`*_mp_original.csv`): If resize was used, coordinates converted back to original dimensions
5. **Log File** (`log_info.txt`): Processing metadata and statistics

## Usage

### GUI Mode (Recommended)

```python
from vaila.markerless_2d_analysis import process_videos_in_directory

# Launch GUI for configuration and processing
process_videos_in_directory()
```

### Programmatic Usage

```python
from vaila.markerless_2d_analysis import process_video

# Process single video with custom configuration
config = {
    'detection_confidence': 0.5,
    'tracking_confidence': 0.5,
    'model_complexity': 1,
    'enable_resize': True,
    'resize_factor': 2,
    'smoothing_algorithm': 'butterworth',
    'temporal_filtering': True
}

process_video('input_video.mp4', 'output_directory', config)
```

## Requirements

- Python 3.12+
- OpenCV (`opencv-python`)
- MediaPipe (`mediapipe`)
- NumPy (`numpy`)
- Pandas (`pandas`)
- SciPy (`scipy`)
- scikit-learn (`scikit-learn`)
- statsmodels (`statsmodels`)
- pykalman (`pykalman`)
- Rich (`rich`) for progress display

## Performance Considerations

- **Memory Usage**: Large videos may require significant RAM, especially with resize enabled
- **Processing Time**: Resize and advanced filtering significantly increase processing time
- **CPU Usage**: Automatic throttling prevents system overload during batch processing

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce batch size or disable resize for large videos
2. **Slow Processing**: Use lower model complexity or disable advanced filtering
3. **Detection Failures**: Adjust confidence thresholds or enable resize for better detection

### Performance Tips

- Use resize only when necessary (for small/distant subjects)
- Choose appropriate model complexity for your hardware
- Process videos in smaller batches for better memory management
- Monitor CPU usage and adjust throttling parameters if needed

## Integration with vailá Ecosystem

This module integrates seamlessly with other vailá tools:

- **Data Processing**: Output coordinates can be used with DLT calibration tools
- **Visualization**: Compatible with vailá's 2D/3D plotting modules
- **Machine Learning**: Processed data can be used for training ML models
- **Multimodal Analysis**: Combine with IMU, force plate, or other sensor data

## Version History

- **v0.6.0**: Added video resize functionality and advanced filtering options
- **v0.5.0**: Added temporal filtering and occlusion estimation
- **v0.4.0**: Added batch processing and memory management
- **v0.3.0**: Added MediaPipe integration with GUI configuration
- **v0.2.0**: Initial implementation with basic pose detection
- **v0.1.0**: Proof of concept with single video processing
