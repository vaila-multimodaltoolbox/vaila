# YOLO Tracking Tools

## Overview

The YOLO Tracking Tools module provides advanced object detection and tracking capabilities using state-of-the-art YOLO models (YOLOv11 and YOLOv12). These tools enable real-time object detection, multi-object tracking, and comprehensive trajectory analysis for biomechanical applications.

## Features

- **Multiple YOLO Versions**: Support for YOLOv11 and YOLOv12 models
- **Real-time Tracking**: Live object detection and tracking with visual feedback
- **Multi-Object Tracking**: Track multiple objects simultaneously with unique IDs
- **Hardware Acceleration**: GPU support for improved performance
- **CSV Export**: Detailed tracking data export for further analysis
- **GUI Interface**: Interactive configuration and visualization
- **ReID Integration**: Advanced re-identification for robust tracking

## Supported YOLO Models

### YOLOv12
- **Latest Generation**: Most advanced YOLO architecture
- **Superior Accuracy**: Improved detection precision
- **Better Speed**: Optimized for real-time applications
- **Enhanced Features**: Advanced object features and embeddings

### YOLOv11
- **Proven Performance**: Well-established architecture
- **Wide Compatibility**: Broad model availability
- **Good Balance**: Speed vs. accuracy optimization
- **Extensive Documentation**: Rich ecosystem support

## Tracking Capabilities

### Object Detection
- **Multiple Classes**: Detect various object types (persons, animals, objects)
- **Confidence Thresholds**: Configurable detection confidence
- **Bounding Boxes**: Precise object localization
- **Feature Extraction**: Rich feature vectors for re-identification

### Multi-Object Tracking
- **Unique ID Assignment**: Persistent tracking across frames
- **Motion Prediction**: Advanced trajectory prediction algorithms
- **Occlusion Handling**: Robust handling of partial occlusions
- **ID Switching Prevention**: Minimize identity switches

### Re-Identification (ReID)
- **Appearance Matching**: Match objects based on visual appearance
- **Multiple Models**: Various ReID architectures (LMBN, OSNet, MobileNet, ResNet, CLIP)
- **Feature Embeddings**: High-dimensional feature representations
- **Cross-Camera Tracking**: Track objects across multiple camera views

## Hardware Acceleration

### GPU Support
- **CUDA**: NVIDIA GPU acceleration for YOLO models
- **Automatic Detection**: Intelligent device selection
- **Performance Optimization**: Maximize throughput on available hardware
- **Memory Management**: Efficient memory usage for large models

### Performance Modes
- **Real-time Mode**: Optimized for live video processing
- **Batch Mode**: Process multiple videos efficiently
- **Accuracy Mode**: Maximum precision for analysis tasks

## Configuration Parameters

### Model Settings
- **Model Selection**: Choose between YOLOv11 and YOLOv12
- **Model Size**: Various model sizes (nano, small, medium, large, extra-large)
- **Confidence Threshold**: Minimum detection confidence (0.0-1.0)
- **IoU Threshold**: Intersection over Union for NMS (0.0-1.0)

### Tracking Settings
- **Tracker Algorithm**: Choose tracking algorithm (BotSort, ByteTrack, etc.)
- **Track Buffer**: Number of frames to keep lost tracks
- **Match Threshold**: Yesilarity threshold for track matching
- **Frame Rate**: Processing frame rate control

### ReID Settings
- **ReID Model**: Select re-identification model architecture
- **Feature Dimension**: Size of feature embeddings
- **Matching Algorithm**: Feature matching strategy
- **Distance Metric**: Metric for similarity calculation

## Output Data

### Tracking Results
- **Bounding Boxes**: Frame-by-frame object locations
- **Track IDs**: Persistent object identifiers
- **Confidence Scores**: Detection confidence for each object
- **Class Labels**: Object class predictions

### CSV Export Format
```csv
Frame,ID,X,Y,Width,Height,Confidence,Class,Features...
1,1,100,150,50,80,0.95,person,0.1,0.2,0.3,...
1,2,200,160,45,75,0.87,person,0.2,0.1,0.4,...
2,1,105,155,50,80,0.93,person,0.11,0.21,0.31,...
```

### Visual Output
- **Annotated Video**: Original video with bounding boxes and track IDs
- **Trajectory Visualization**: Object movement paths
- **Heat Maps**: Areas of high object activity
- **Statistics Overlay**: Real-time tracking statistics

## Usage

### GUI Mode (Recommended)

```python
from vaila.yolov12track import run_yolov12track
from vaila.yolov11track import run_yolov11track

# Launch YOLOv12 tracking interface
run_yolov12track()

# Launch YOLOv11 tracking interface
run_yolov11track()
```

### Programmatic Usage

```python
from vaila.yolov12track import process_video
from ultralytics import YOLO

# Load YOLO model
model = YOLO('yolov12n.pt')  # or yolov12s.pt, yolov12m.pt, etc.

# Process video with custom settings
process_video(
    input_file='input_video.mp4',
    output_dir='/path/to/output',
    model=model,
    confidence=0.5,
    track=True,
    show=True
)
```

### Batch Processing

```python
import glob
from vaila.yolov12track import process_video

# Process multiple videos
video_files = glob.glob('/path/to/videos/*.mp4')

for video_file in video_files:
    process_video(
        input_file=video_file,
        output_dir='/path/to/tracking_results',
        confidence=0.6,
        track_buffer=30
    )
```

## Advanced Configuration

### Custom Model Loading

```python
from ultralytics import YOLO

# Load custom trained model
custom_model = YOLO('/path/to/custom_model.pt')

# Process with custom model
process_video(
    input_file='test_video.mp4',
    model=custom_model,
    output_dir='/path/to/results'
)
```

### Multi-Camera Tracking

```python
# Configure for multi-camera setup
tracking_config = {
    'camera_matrix': camera_calibration_matrix,
    'distortion_coeffs': distortion_coefficients,
    'camera_positions': camera_positions_list,
    'cross_camera_matching': True
}

process_video(
    input_file='multi_camera_video.mp4',
    tracking_config=tracking_config
)
```

## Performance Optimization

### Speed vs. Accuracy Trade-offs

| Model Size | Speed (FPS) | Accuracy | Memory Usage | Use Case |
|------------|-------------|----------|--------------|----------|
| Nano (n) | 50-100 | Good | Low | Real-time |
| Small (s) | 30-60 | Better | Medium | Balanced |
| Medium (m) | 15-30 | High | High | Analysis |
| Large (l) | 8-15 | Very High | Very High | Maximum accuracy |
| Extra (x) | 3-8 | Maximum | Extreme | Research |

### Hardware Recommendations

- **GPU Memory**: 4-8GB for real-time processing
- **CPU**: Multi-core for batch processing
- **RAM**: 16GB+ for large models and datasets
- **Storage**: SSD for fast video I/O

### Optimization Techniques

- **Model Quantization**: Reduce model size and inference time
- **Batch Processing**: Process multiple frames simultaneously
- **Frame Skipping**: Process every Nth frame for speed
- **Resolution Reduction**: Downscale input for faster processing

## Integration with vailá Ecosystem

This module integrates with other vailá tools:

- **Markerless Analysis**: Use tracking data for pose estimation
- **Video Processing**: Combine with video compression and editing tools
- **Data Analysis**: Export tracking data for statistical analysis
- **Visualization**: Create trajectory visualizations and heat maps

## Applications

### Biomechanical Analysis
- **Gait Analysis**: Track limb movements during walking/running
- **Sports Performance**: Analyze athlete movements and techniques
- **Rehabilitation**: Monitor patient progress and movement quality
- **Animal Behavior**: Track animal movements in research settings

### Surveillance and Security
- **People Tracking**: Monitor pedestrian movement patterns
- **Crowd Analysis**: Analyze crowd density and flow
- **Security Monitoring**: Detect and track suspicious activities
- **Traffic Analysis**: Monitor vehicle and pedestrian traffic

### Research Applications
- **Behavioral Studies**: Analyze social interaction patterns
- **Ecological Research**: Track animal behavior in natural habitats
- **Industrial Monitoring**: Quality control and process monitoring

## Troubleshooting

### Common Issues

1. **Model Loading Errors**:
   - Ensure model files are in the correct path
   - Check internet connection for automatic downloads
   - Verify model compatibility with YOLO version

2. **Performance Issues**:
   - Enable GPU acceleration if available
   - Reduce model size for faster processing
   - Lower input resolution for speed improvement

3. **Tracking Problems**:
   - Adjust confidence thresholds for better detection
   - Increase track buffer for handling occlusions
   - Fine-tune ReID model for better identity preservation

4. **Memory Issues**:
   - Process videos in smaller batches
   - Use smaller model sizes for limited memory
   - Clear GPU memory between processing tasks

### Debugging Tools

- **Verbose Logging**: Enable detailed logging for troubleshooting
- **Visualization**: Use GUI to preview detection results
- **Performance Monitoring**: Track FPS and memory usage
- **Error Reporting**: Comprehensive errorrr messages and warnings

## Version History

### YOLOv12 Module
- **v0.0.3**: Added ReID integration and improved tracking algorithms
- **v0.0.2**: Added multi-model support and hardware detection
- **v0.0.1**: Initial implementation with basic YOLOv12 tracking

### YOLOv11 Module
- **v1.0**: Full implementation with advanced tracking features

## Requirements

### Core Dependencies
- **Python 3.8+**: Modern Python features and performance
- **PyTorch**: Deep learning framework for YOLO models
- **Ultralytics**: YOLO model implementation
- **OpenCV**: Computer vision operations
- **NumPy**: Numerical computations

### Optional Dependencies
- **CUDA Toolkit**: For GPU acceleration (NVIDIA GPUs)
- **BoxMOT**: Advanced tracking algorithms
- **FFmpeg**: Video format conversion

### Installation

#### Core Installation
```bash
pip install torch torchvision ultralytics opencv-python numpy
```

#### GPU Support (Optional)
```bash
# NVIDIA GPUs
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### Advanced Tracking
```bash
pip install boxmot  # For advanced tracking algorithms
```

## References

- **YOLOv12 Paper**: Latest YOLO architecture developments
- **Ultralytics Documentation**: Official YOLO implementation guide
- **Computer Vision**: Object detection and tracking fundamentals
- **Deep Learning**: Neural network architectures for vision tasks
