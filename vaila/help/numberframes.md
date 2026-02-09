# numberframes.py

## 📋 Module Information

- **Category:** Tools
- **File:** `vaila/numberframes.py`
- **Version:** 0.1.4
- **Author:** Paulo R. P. Santiago
- **Email:** paulosantiago@usp.br
- **GitHub:** https://github.com/vaila-multimodaltoolbox/vaila
- **GUI Interface:** ✅ Yes
- **License:** GNU Lesser General Public License v3.0

## 📖 Description

This script provides scientific-grade video metadata extraction with high precision for research applications. It analyzes video files within a selected directory and extracts comprehensive metadata including frame count, frame rates (display and capture), resolution, codec information, and duration. The script generates both a user-friendly summary and detailed JSON metadata files with precision suitable for scientific research.

### Key Features

- **Scientific Precision**: Frame rates and durations with 9 decimal places for research accuracy
- **Fast Extraction**: Single `ffprobe` JSON call for efficient metadata extraction
- **Capture FPS Detection**: Automatically detects Android slow-motion capture FPS tags
- **Parallel Processing**: Processes multiple videos concurrently for faster analysis
- **Dual Output**: Basic summary file and full JSON metadata files
- **GUI Display**: User-friendly table view of all video metadata
- **Research Ready**: High-precision values suitable for scientific analysis

## 🔧 Main Functions

**Total functions found:** 10+

### Core Functions

- `get_video_info()` - Extract comprehensive video metadata using ffprobe
- `_ffprobe_json()` - Single fast ffprobe call returning complete JSON
- `_to_float_fps()` - Convert fraction strings to precise float values
- `_extract_capture_fps()` - Detect Android capture FPS from metadata tags

### Display and Output Functions

- `display_video_info()` - GUI table displaying all video metadata
- `save_basic_metadata_to_file()` - Save human-readable summary (9 decimal precision)
- `save_full_metadata_to_file()` - Save complete JSON metadata for each video
- `show_save_success_message()` - Confirmation dialog

### Main Entry Point

- `count_frames_in_videos()` - Main function with directory selection and processing

## 📊 Metadata Extracted

### Frame Information

- **Frame Count**: Exact number of frames (from `nb_frames` or calculated)
- **Display FPS**: Frame rate for display (r_frame_rate) - 9 decimal precision
- **Average FPS**: Average frame rate (avg_frame_rate) - 9 decimal precision
- **Capture FPS**: Real capture rate for slow-motion videos (Android tags) - 9 decimal precision
- **Recommended Sampling Hz**: Best sampling rate for data analysis

### Video Properties

- **Resolution**: Width x Height in pixels
- **Duration**: Video length in seconds (9 decimal precision)
- **Codec**: Video codec name and long name
- **Container**: Container format name and long name

### Slow-Motion Detection

- Automatically detects `com.android.capture.fps` tags
- Calculates slow-motion factor (capture_fps / display_fps)
- Provides recommended sampling rate for analysis

## 🔬 Scientific Precision

### High-Precision Values

All frame rates and durations are stored and displayed with **9 decimal places** for scientific accuracy:

- **Display FPS**: `59.940059940` (instead of rounded `59.940`)
- **Duration**: `253.620000000` seconds (instead of `253.62`)
- **Capture FPS**: `240.000000000` for slow-motion videos

### Fraction-to-Float Conversion

The script converts frame rate fractions (e.g., "30000/1001") to precise float values:

```python
# Example: 30000/1001 = 29.97002997002997 fps
# Preserved exactly, not rounded to 29.97
```

### Frame Count Calculation

1. **Primary**: Uses `nb_frames` from video stream if available
2. **Secondary**: Calculates from `duration × avg_frame_rate` if `nb_frames` missing
3. **Fallback**: Uses OpenCV frame count if ffprobe unavailable

## 📁 Output Files

### Basic Metadata File

**File**: `video_metadata_basic_TIMESTAMP.txt`

Contains human-readable summary with high precision:

```
File: video.mp4
Frames: 15202
Display_FPS: 59.940059940
Avg_FPS: 59.940059940
Capture_FPS: 240.000000000
Recommended_Sampling_Hz: 240.000000000
SlowMo_Factor: 4.003996004
Codec: hevc - H.265 / HEVC (High Efficiency Video Coding)
Container: mov,mp4,m4a,3gp,3g2,mj2 - QuickTime / MOV
Resolution: 1920x1080
Duration (s): 253.620000000
```

### Full Metadata Files

**Directory**: `metadata_full_TIMESTAMP/`

Contains complete JSON metadata for each video:

- `video_name.json` - Full ffprobe output with all streams, format, chapters, etc.
- Suitable for programmatic analysis
- Includes all technical details

## 🚀 Usage

### GUI Mode (Recommended)

```python
from vaila.numberframes import count_frames_in_videos

# Launch GUI for directory selection and analysis
count_frames_in_videos()
```

### Programmatic Usage

```python
from vaila.numberframes import get_video_info
from pathlib import Path

# Get precise metadata for a single video
video_path = Path("video.mp4")
info = get_video_info(str(video_path))

# Access high-precision values
print(f"Display FPS: {info['display_fps']:.9f}")
print(f"Capture FPS: {info['capture_fps']:.9f}" if info['capture_fps'] else "N/A")
print(f"Duration: {info['duration']:.9f} seconds")
print(f"Frames: {info['frame_count']}")
print(f"Resolution: {info['resolution']}")
```

## 💻 Requirements

- **Python**: 3.12.11+
- **ffprobe**: Required for metadata extraction (part of ffmpeg)
- **Tkinter**: For GUI interface (usually included with Python)
- **rich**: For enhanced console output

### Installing ffmpeg/ffprobe

- **Windows**: Download from https://ffmpeg.org/download.html
- **macOS**: `brew install ffmpeg`
- **Linux**: `sudo apt-get install ffmpeg` (Ubuntu/Debian)

## ⚙️ Technical Details

### Parallel Processing

- Uses `ThreadPoolExecutor` for concurrent video analysis
- Default workers: `max(2, cpu_count())`
- Processes videos in parallel for faster batch analysis

### Frame Rate Extraction

1. Extracts `r_frame_rate` (rational frame rate) as fraction string
2. Extracts `avg_frame_rate` (average frame rate) as fraction string
3. Converts fractions to float with full precision
4. Prefers `avg_frame_rate` if available, falls back to `r_frame_rate`

### Capture FPS Detection

Searches for Android slow-motion tags in:

- Format tags: `com.android.capture.fps`
- Stream tags: `com.android.capture.fps`
- Alternative tags: `com.android.capturer.fps`, `com.android.slowMotion.capture.fps`

### Error Handling

- Graceful fallback if ffprobe fails
- Continues processing other videos if one fails
- Detailed error messages in output files

## 🔬 Scientific Research Applications

This tool is essential for research requiring precise video metadata:

- **Biomechanics**: Accurate frame rates for temporal analysis
- **Motion Capture**: Frame-accurate synchronization
- **Video Analysis**: Precise duration and frame count for calculations
- **Data Validation**: Verify video properties before processing
- **Slow-Motion Analysis**: Detect and account for slow-motion capture rates

## 📊 Example Output

### Console Output

```
Video info: 1920x1080, codec=hevc, container=mov,mp4,m4a,3gp,3g2,mj2,
display≈59.940059940 fps, avg≈59.940059940 fps, cap=240.000000000 Hz,
dur=253.620000000s, frames=15202
```

### GUI Display

Shows table with:

- Video file names
- Frame counts
- Display FPS (6 decimals for readability)
- Capture FPS (6 decimals)
- Codec and container
- Resolution
- Duration (6 decimals)

### File Output

All values saved with 9 decimal precision for maximum accuracy.

## 📝 Notes

- **Precision**: All FPS and duration values use 9 decimal places in files
- **GUI Display**: Shows 6 decimal places for readability (full precision in files)
- **Frame Count**: Uses `nb_frames` when available, otherwise calculated
- **Slow-Motion**: Automatically detects and reports slow-motion factors
- **Batch Processing**: Processes all videos in selected directory

## 🐛 Troubleshooting

### ffprobe Not Found

- Install ffmpeg package (includes ffprobe)
- Ensure ffprobe is in system PATH
- Script will show error but continue with other videos

### Missing Frame Count

- Some video formats don't include `nb_frames`
- Script calculates from duration × FPS
- Result may differ slightly from actual frame count

### Slow-Motion Not Detected

- Only detects Android-specific tags
- Other slow-motion formats may not be detected
- Check full JSON metadata for other tags

## 📚 Related Tools

- **cutvideo.py**: Cut videos with scientific precision
- **syncvid.py**: Synchronize multiple videos
- **resize_video.py**: Resize videos while preserving metadata

---

📅 **Last Updated:** January 2025  
🔗 **Part of vailá - Multimodal Toolbox**  
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
