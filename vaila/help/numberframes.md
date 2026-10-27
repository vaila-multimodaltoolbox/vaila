# numberframes.py

## 📋 Module Information

- **Category:** Tools
- **File:** `vaila/numberframes.py`
- **Version:** 0.4.12
- **Updated:** 27 October 2026
- **Author:** Paulo R. P. Santiago
- **Email:** paulosantiago@usp.br
- **GitHub:** https://github.com/vaila-multimodaltoolbox/vaila
- **GUI Interface:** Yes (*vailá* Frame C → **Metadata info**)
- **License:** GNU Lesser General Public License v3.0

## Description

This script provides scientific-grade video metadata extraction with high precision for research applications. It analyzes video files within a selected directory and extracts comprehensive metadata including frame count, frame rates (display and capture), resolution, codec information, and duration.

It also **Adjusts real FPS**: retags the container frame rate (stream-copy remux, re-encode fallback) when videos were written with a stamped FPS different from the capture rate (common after joining clips). Cut Video and Make Sync file both read **container** FPS, so fix the files here first.

### Key Features

- **Scientific Precision**: Frame rates and durations with 9 decimal places for research accuracy
- **Fast Extraction**: Single `ffprobe` JSON call for efficient metadata extraction
- **Parallel Processing**: Processes multiple videos concurrently for faster analysis
- **Dual Output**: Basic summary file and full JSON metadata files
- **Adjust real FPS**: Remux (`setts` + MP4 timescale) keeping every frame; duration becomes `frames / real_fps`
- **Research Ready**: High-precision values suitable for scientific analysis

## Main Functions

### Metadata

- `get_precise_video_metadata()` — ffprobe JSON → fps, frames, resolution, codec
- `get_video_info()` — summary dict used by the batch extractor
- `count_frames_in_videos()` — GUI/CLI metadata extraction entry point

### Adjust real FPS

- `parse_fps_value()` / `parse_fps_fraction()` — float or `num/den`
- `adjust_video_fps(src, real_fps, out_path)` — one file (copy, then re-encode fallback)
- `adjust_videos_fps(...)` — batch → `processed_fpsfix_<timestamp>/` + `fpsfix_summary.csv`

## Adjust real FPS

### When to use

Joined or re-muxed clips often keep a **wrong stamped FPS** while every frame is still at the capture rate. Typing “real FPS” in Cut/Sync only helps math when it **matches** the container; otherwise fix the file here so Cut Video seeks/writes at the capture rate automatically.

### GUI

1. Frame C → Video → **Metadata info**
2. Choose **Adjust real FPS**
3. Enter real FPS once for the batch (`29.97` or `30000/1001`)
4. Select the directory of videos

### CLI

```bash
uv run vaila/numberframes.py --set-fps 29.97 --dir /path/to/videos --cli
uv run vaila/numberframes.py --set-fps 30000/1001 video1.mp4 video2.mp4 --cli
```

### Output

- Directory: `processed_fpsfix_YYYYMMDD_HHMMSS/` next to the sources
- Remuxed videos (same filenames)
- `fpsfix_summary.csv` — old/new fps, frames, method (`copy`|`reencode`), errors

Semantics: **keeps every frame**; does not drop/duplicate to stretch playback. Reported duration becomes `nb_frames / real_fps`.

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

📅 **Last Updated:** 25 August 2026
🔗 **Part of vailá - Multimodal Toolbox**  
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
