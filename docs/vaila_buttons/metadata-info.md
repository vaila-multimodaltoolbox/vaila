# Metadata Info Tool

## Overview

The **Metadata Info** tool (`numberframes.py`) provides **scientific-grade video metadata extraction** with high precision for research applications. It analyzes video files within a selected directory and extracts comprehensive metadata including frame count, frame rates (display and capture), resolution, codec information, and duration.

## Key Features

- **Scientific Precision**: Frame rates and durations with 9 decimal places for research accuracy
- **Fast Extraction**: Single `ffprobe` JSON call for efficient metadata extraction
- **Capture FPS Detection**: Automatically detects Android slow-motion capture FPS tags
- **Parallel Processing**: Processes multiple videos concurrently for faster analysis
- **Dual Output**: Basic summary file and full JSON metadata files
- **GUI Display**: User-friendly table view of all video metadata
- **Research Ready**: High-precision values suitable for scientific analysis

## How to Use

### Accessing the Tool

1. Launch vailá GUI (`vaila.py`)
2. Navigate to **Tools** section
3. Click **Metadata Info** button
4. Select directory containing video files

### Output Files

#### Basic Metadata File
**File**: `video_metadata_basic_TIMESTAMP.txt`

Contains human-readable summary with high precision:
- Frame count
- Display FPS (9 decimal places)
- Average FPS (9 decimal places)
- Capture FPS (9 decimal places) - for slow-motion videos
- Recommended sampling rate
- Codec and container information
- Resolution
- Duration (9 decimal places)

#### Full Metadata Files
**Directory**: `metadata_full_TIMESTAMP/`

Contains complete JSON metadata for each video:
- Full ffprobe output
- All streams, format, chapters
- Suitable for programmatic analysis

## Scientific Precision

### High-Precision Values
All frame rates and durations are stored with **9 decimal places**:
- **Display FPS**: `59.940059940` (instead of rounded `59.940`)
- **Duration**: `253.620000000` seconds (instead of `253.62`)
- **Capture FPS**: `240.000000000` for slow-motion videos

### Frame Rate Extraction
- Extracts frame rates as fraction strings (e.g., "30000/1001")
- Converts to precise float values
- Preserves exact frame rates without rounding

## Metadata Extracted

### Frame Information
- Frame count (from `nb_frames` or calculated)
- Display FPS (r_frame_rate)
- Average FPS (avg_frame_rate)
- Capture FPS (Android tags for slow-motion)
- Recommended sampling rate

### Video Properties
- Resolution (width × height)
- Duration (seconds)
- Codec name and description
- Container format and description

### Slow-Motion Detection
- Automatically detects `com.android.capture.fps` tags
- Calculates slow-motion factor
- Provides recommended sampling rate

## Requirements

- **ffprobe**: Required for metadata extraction (part of ffmpeg)
- **Tkinter**: For GUI interface
- **rich**: For enhanced console output

## Use Cases

- **Biomechanics**: Accurate frame rates for temporal analysis
- **Motion Capture**: Frame-accurate synchronization
- **Video Analysis**: Precise duration and frame count for calculations
- **Data Validation**: Verify video properties before processing
- **Slow-Motion Analysis**: Detect and account for slow-motion capture rates

## Related Tools

- **Cut Videos** (`cutvideo.py`): Cut videos with scientific precision
- **Sync Videos** (`syncvid.py`): Synchronize multiple videos
- **Resize Video** (`resize_video.py`): Resize videos while preserving metadata

## Documentation

For complete documentation, see:
- [Full Help Documentation](../../vaila/help/numberframes.html)
- [Markdown Documentation](../../vaila/help/numberframes.md)
