# Video Compression Tools

## Overview

The Video Compression Tools module provides comprehensive video compression capabilities using multiple codecs (H.264, H.265, H.266) with hardware acceleration support. These tools enable efficient video size reduction while maintaining quality for biomechanical analysis workflows.

## Features

- **Multiple Codec Support**: H.264 (AVC), H.265 (HEVC), and H.266 (VVC) compression
- **Hardware Acceleration**: Automatic detection and use of GPU acceleration (NVIDIA NVENC, Intel Quick Sync, AMD AMF)
- **Batch Processing**: Compress multiple videos simultaneously
- **Quality Control**: Configurable quality settings with Constant Rate Factor (CRF)
- **Resolution Options**: Support for various output resolutions
- **GUI Interface**: User-friendly interface for easy configuration
- **Cross-Platform**: Support for Windows, Linux, and macOS

## Supported Codecs

### H.264 (AVC) - Advanced Video Coding
- **Compatibility**: Universal support across all devices and platforms
- **Compression**: Good balance between file size and quality
- **Speed**: Fast encoding, especially with hardware acceleration
- **Use Case**: General purpose compression, web distribution

### H.265 (HEVC) - High Efficiency Video Coding
- **Compression**: Significantly better compression than H.264 (up to 50% size reduction)
- **Quality**: Maintains high quality at lower bitrates
- **Speed**: Slower encoding but better results
- **Use Case**: High-quality archival, limited storage scenarios

### H.266 (VVC) - Versatile Video Coding
- **Compression**: Most advanced codec with up to 50% better compression than H.265
- **Quality**: Superior quality preservation at very low bitrates
- **Speed**: Very slow encoding, requires significant processing power
- **Use Case**: Future-proof archival, maximum compression scenarios

## Hardware Acceleration

### NVIDIA NVENC
- **Supported GPUs**: Kepler architecture and newer (GTX 650+)
- **Performance**: Up to 10x faster encoding
- **Quality**: Excellent quality with hardware optimization
- **Requirements**: NVIDIA drivers with NVENC support

### Intel Quick Sync Video
- **Supported CPUs**: 3rd generation Intel Core processors and newer
- **Performance**: Good acceleration for integrated graphics
- **Compatibility**: Built into Intel HD Graphics and Iris Graphics

### AMD AMF (Advanced Media Framework)
- **Supported GPUs**: Radeon HD 7000 series and newer
- **Performance**: Significant speedup for AMD graphics cards
- **Features**: Hardware-accelerated H.264 and H.265 encoding

## Configuration Parameters

### Quality Settings
- **Constant Rate Factor (CRF)**: Quality-based encoding (0-51 scale)
  - Lower values = higher quality, larger files
  - Higher values = lower quality, smaller files
  - Recommended range: 18-28 for good quality
- **Preset**: Encoding speed vs. compression efficiency trade-off
  - `ultrafast` to `veryslow`: Slower presets provide better compression

### Resolution Options
- **Original Resolution**: Maintain input resolution
- **Custom Resolution**: Specify exact output dimensions
- **Scaling**: Automatic downscaling for size reduction

### Performance Settings
- **Hardware Encoder Selection**: Automatic or manual selection
- **CPU Threads**: Configurable thread count for software encoding
- **Batch Size**: Number of simultaneous encoding processes

## Usage

### GUI Mode (Recommended)

```python
from vaila.compress_videos_h264 import compress_videos_h264_gui
from vaila.compress_videos_h265 import compress_videos_h265_gui
from vaila.compress_videos_h266 import compress_videos_h266_gui

# Launch compression interface for different codecs
compress_videos_h264_gui()  # H.264 compression
compress_videos_h265_gui()  # H.265 compression
compress_videos_h266_gui()  # H.266 compression
```

### Programmatic Usage

```python
from vaila.compress_videos_h264 import run_compress_videos_h264

# Configure compression parameters
compression_config = {
    'preset': 'fast',      # Encoding speed preset
    'crf': 23,            # Quality setting (18-28 recommended)
    'resolution': '1080p', # Output resolution
    'use_gpu': True       # Enable hardware acceleration
}

# Compress videos
run_compress_videos_h264(
    input_list=['video1.mp4', 'video2.mp4'],
    output_dir='/path/to/compressed',
    **compression_config
)
```

### Batch Processing

```python
from vaila.compress_videos_h264 import find_videos

# Find all videos in directory
video_files = find_videos('/path/to/videos')

# Compress all videos
for video_file in video_files:
    run_compress_videos_h264(
        input_list=[video_file],
        output_dir='/path/to/compressed',
        preset='medium',
        crf=25,
        use_gpu=True
    )
```

## Advanced Configuration

### Custom FFmpeg Parameters

```python
# Advanced FFmpeg options for expert users
advanced_options = [
    '-tune', 'film',           # Tune for film content
    '-profile:v', 'main',      # H.264 profile
    '-pix_fmt', 'yuv420p',     # Pixel format for compatibility
    '-movflags', '+faststart'  # Enable streaming
]

run_compress_videos_h264(
    input_list=videos,
    output_dir=output_dir,
    preset='slow',
    crf=20,
    custom_ffmpeg_args=advanced_options
)
```

### Multi-Pass Encoding

```python
# Two-pass encoding for optimal quality
run_compress_videos_h265(
    input_list=videos,
    output_dir=output_dir,
    two_pass=True,      # Enable two-pass encoding
    target_bitrate='5M' # Target bitrate for second pass
)
```

## Performance Optimization

### Encoding Speed vs. Quality

| Preset | Speed | Quality | Use Case |
|--------|-------|---------|----------|
| ultrafast | Very Fast | Lower | Quick previews |
| superfast | Fast | Good | Fast compression |
| veryfast | Fast | Good | General use |
| faster | Medium | Better | Good quality |
| fast | Medium | Better | Balanced |
| medium | Slow | Best | Best quality |
| slow | Very Slow | Excellent | Maximum quality |
| slower | Very Slow | Excellent | Archive quality |
| veryslow | Extremely Slow | Maximum | Best possible |

### Hardware Acceleration Benefits

- **Speed Improvement**: 3-10x faster encoding
- **Power Efficiency**: Lower CPU usage and power consumption
- **Quality**: Often better quality at same bitrate
- **Concurrent Tasks**: Enable multiple compression jobs

### Memory Management

- **Batch Size**: Process fewer videos simultaneously for limited RAM
- **Resolution Limits**: Consider downscaling very high-resolution videos
- **Chunked Processing**: Process long videos in segments

## Quality Assurance

### Compression Quality Checks

- **File Size Reduction**: Monitor compression ratio vs. quality loss
- **Visual Quality**: Compare compressed vs. original videos
- **Artifact Detection**: Check for compression artifacts (blocking, banding)
- **Playback Compatibility**: Verify compressed videos play on target devices

### Performance Metrics

- **Encoding Speed**: Track compression speed in FPS or MB/s
- **Quality Metrics**: Use PSNR, SSIM for objective quality assessment
- **File Size**: Monitor achieved compression ratios

## Integration with vailá Ecosystem

This module integrates with other vailá tools:

- **Video Processing**: Use compressed videos for further analysis
- **Markerless Analysis**: Compress videos before pose estimation
- **Data Storage**: Reduce storage requirements for video datasets
- **Distribution**: Prepare videos for sharing and presentation

## Troubleshooting

### Common Issues

1. **Hardware Acceleration Not Working**:
   - Verify GPU drivers are installed and up to date
   - Check FFmpeg supports the hardware encoder
   - Ensure GPU is not heavily utilized by other applications

2. **Poor Compression Results**:
   - Lower CRF value for better quality (but larger files)
   - Use slower preset for better compression efficiency
   - Consider two-pass encoding for optimal results

3. **Long Encoding Times**:
   - Enable hardware acceleration if available
   - Use faster preset (may reduce quality)
   - Process videos in smaller batches

4. **File Compatibility Issues**:
   - Use standard pixel format (`yuv420p`) for broad compatibility
   - Avoid experimental codecs for production use
   - Test playback on target devices

### Performance Tuning

- **Optimal CRF Values**:
  - CRF 18-22: Visually lossless quality
  - CRF 23-28: Good quality for analysis
  - CRF 29-32: Acceptable for preview purposes

- **Hardware Selection**:
  - Use NVIDIA GPUs for best H.264/H.265 performance
  - Intel Quick Sync provides good performance for integrated graphics
  - AMD GPUs offer competitive performance for newer models

## Version History

### H.264 Module
- **v2.0**: Added hardware acceleration and advanced GUI options
- **v1.5**: Added batch processing and quality presets
- **v1.0**: Initial implementation with basic compression

### H.265 Module
- **v1.0**: Full implementation with hardware acceleration support

### H.266 Module
- **v1.0**: Initial implementation with experimental VVC support

## Requirements

### Core Dependencies
- **FFmpeg**: Required for all compression operations
- **Python 3.8+**: Modern Python features and performance
- **Tkinter**: For GUI interface (usually included)

### Hardware Acceleration (Optional)
- **NVIDIA Drivers**: For NVENC acceleration (Windows/Linux)
- **Intel Graphics Drivers**: For Quick Sync (Windows/Linux)
- **AMD Drivers**: For AMF acceleration (Windows/Linux)

### Installation

#### FFmpeg Installation
```bash
# Conda (recommended)
conda install -c conda-forge ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

#### Hardware Acceleration Setup

**NVIDIA (Windows/Linux)**:
```bash
# Verify NVENC support
ffmpeg -encoders | grep nvenc

# Test encoding
ffmpeg -f lavfi -i color=black:s=320x240:r=1:d=1 -c:v h264_nvenc -f null -
```

**Intel Quick Sync**:
```bash
# Verify Quick Sync support
ffmpeg -encoders | grep qsv

# Test encoding
ffmpeg -f lavfi -i color=black:s=320x240:r=1:d=1 -c:v h264_qsv -f null -
```

## References

- **FFmpeg Documentation**: Official FFmpeg encoding guide
- **Video Codecs**: H.264, H.265, H.266 specification documents
- **Hardware Acceleration**: GPU encoding best practices
- **Quality Assessment**: PSNR and SSIM measurement techniques
