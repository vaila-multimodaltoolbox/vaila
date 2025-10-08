# Video Processing Tools

## Overview

The Video Processing Tools module provides comprehensive video manipulation capabilities for biomechanical analysis, including merging, splitting, format conversion, and batch processing operations.

## Features

- **Video Merging**: Combine videos with their reversed versions for loop creation
- **Video Splitting**: Split videos at specified time points or into halves
- **Batch Processing**: Process multiple videos simultaneously
- **Hardware Acceleration**: Automatic detection and use of GPU acceleration
- **Format Conversion**: Convert between various video formats
- **Custom Processing Lists**: Use text files to specify custom processing operations
- **Progress Tracking**: Real-time progress monitoring with detailed logging

## Core Operations

### 1. Video Merging
Create videos with doubled frame count by merging original and reversed versions:

- **Forward-Backward Loop**: Combine video with its time-reversed version
- **Seamless Looping**: Create smooth transitions between forward and backward playback
- **Custom Transitions**: Optional crossfade effects between segments

### 2. Video Splitting
Split videos at specified time points or into equal segments:

- **Time-based Splitting**: Split at specific timestamps
- **Half Splitting**: Split each video into first and second halves
- **Custom Segmentation**: Define custom split points and segments

### 3. Batch Processing
Process multiple videos with custom instructions:

- **Text File Control**: Use configuration files for complex batch operations
- **Directory Processing**: Process all videos in specified directories
- **Custom Naming**: Automatic output file naming with timestamps

## Supported Video Formats

### Input Formats
- **MP4**: H.264/H.265 encoded MP4 files
- **AVI**: Audio Video Interleave format
- **MOV**: QuickTime movie format
- **MKV**: Matroska video container
- **WebM**: WebM format with VP8/VP9 encoding

### Output Formats
- **MP4**: H.264/H.265 compressed MP4 (recommended)
- **AVI**: Uncompressed or compressed AVI
- **MOV**: QuickTime compatible format
- **Same as Input**: Preserve original format when possible

## Hardware Acceleration

The module automatically detects and utilizes available hardware acceleration:

### GPU Support
- **NVIDIA NVENC**: H.264/H.265 encoding with NVIDIA GPUs
- **Intel Quick Sync**: Hardware encoding with Intel integrated graphics
- **AMD AMF**: Accelerated encoding with AMD GPUs

### Performance Optimization
- **Automatic Detection**: Tests hardware encoders for compatibility
- **Fallback Strategy**: Falls back to software encoding if hardware fails
- **Quality Preservation**: Maintains video quality across encoding methods

## Configuration Files

### Processing List Format
Create text files with custom processing instructions:

```txt
# videos_e_frames.txt
video1.mp4,merge
video2.mp4,split,00:00:30
video3.mp4,merge,00:01:00,00:02:00
```

### Configuration Parameters
- **Operation Type**: `merge`, `split`, or `convert`
- **Time Points**: Specify split times in HH:MM:SS format
- **Output Options**: Quality settings and format preferences

## Usage

### GUI Mode (Recommended)

```python
from vaila.videoprocessor import process_videos_gui

# Launch GUI for interactive processing
process_videos_gui()
```

### Programmatic Usage

```python
from vaila.videoprocessor import process_videos_merge, process_videos_split

# Merge videos with reversed versions
process_videos_merge(
    source_directory='/path/to/videos',
    output_directory='/path/to/output',
    file_list=['video1.mp4', 'video2.mp4']
)

# Split videos into halves
process_videos_split(
    source_directory='/path/to/videos',
    output_directory='/path/to/output',
    split_type='half'  # or 'time' for time-based splitting
)
```

### Batch Processing with Configuration

```python
from vaila.videoprocessor import process_videos_gui

# Process using configuration file
process_videos_gui(
    config_file='/path/to/processing_list.txt',
    output_directory='/path/to/output'
)
```

## Advanced Configuration

### Quality Settings
- **Compression Quality**: Control output file size vs. quality trade-off
- **Resolution**: Maintain original resolution or resize if needed
- **Frame Rate**: Preserve or modify frame rates
- **Bitrate Control**: Set target bitrates for consistent file sizes

### Audio Processing
- **Audio Preservation**: Maintain original audio tracks when possible
- **Audio Removal**: Option to remove audio for smaller file sizes
- **Audio Format**: Convert between different audio codecs

### Output Organization
- **Timestamp Directories**: Automatic creation of timestamped output folders
- **Original Structure**: Preserve directory structure in output
- **Custom Naming**: Configurable output file naming schemes

## Performance Considerations

### Processing Speed
- **Hardware Acceleration**: 3-10x faster with GPU support
- **Batch Size**: Optimal batch sizes depend on system memory
- **Video Resolution**: Higher resolution significantly impacts processing time

### Memory Usage
- **Large Videos**: Consider processing in segments for 4K+ videos
- **Multiple Files**: Batch processing may require substantial RAM
- **Temporary Files**: FFmpeg may create temporary files during processing

### Storage Requirements
- **Output Size**: Merged videos are approximately 2x original size
- **Working Space**: Temporary files may double storage needs during processing
- **Cleanup**: Automatic cleanup of temporary files after processing

## Quality Assurance

### Video Quality Checks
- **Frame Count Verification**: Ensure all frames are processed
- **Resolution Validation**: Verify output resolution matches expectations
- **Audio Synchronization**: Check audio-video synchronization

### Error Handling
- **Corrupted Files**: Graceful handling of corrupted video files
- **Missing Codecs**: Automatic fallback to compatible codecs
- **Disk Space**: Monitoring and warnings for insufficient storage

## Integration with vailá Ecosystem

This module integrates with other vailá tools:

- **Markerless Analysis**: Process videos before pose estimation
- **Visualization**: Create videos for presentation and analysis
- **Compression Tools**: Use with video compression modules
- **Synchronization**: Combine with video synchronization tools

## Troubleshooting

### Common Issues

1. **FFmpeg Not Found**: Install FFmpeg and ensure it's in system PATH
2. **Hardware Acceleration Issues**: Verify GPU drivers and codec support
3. **Memory Errors**: Process large videos individually or increase system memory
4. **Audio Issues**: Check audio codec compatibility and format support

### Performance Optimization

- **Use Hardware Acceleration**: Enable GPU encoding when available
- **Batch Size Optimization**: Find optimal batch sizes for your system
- **Quality vs. Speed**: Balance compression settings for your needs
- **Parallel Processing**: Utilize multiple CPU cores when possible

### File Format Issues

- **Unsupported Codecs**: Convert to standard formats before processing
- **Corrupted Files**: Verify file integrity before processing
- **Large Files**: Consider splitting very large files before batch processing

## Version History

- **v2.0**: Added hardware acceleration and advanced configuration options
- **v1.5**: Added batch processing with configuration files
- **v1.0**: Initial implementation with basic merge and split operations

## Requirements

### Core Dependencies
- **FFmpeg**: Required for all video processing operations
- **Python 3.8+**: Modern Python features and performance
- **Tkinter**: For GUI components (usually included)

### Optional Dependencies
- **Hardware Acceleration**: NVIDIA/AMD/Intel GPU drivers for hardware encoding
- **Rich**: Enhanced console output (optional)

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

## References

- **FFmpeg Documentation**: Official FFmpeg documentation and examples
- **Video Processing**: Standards for digital video processing
- **Hardware Acceleration**: GPU encoding best practices
