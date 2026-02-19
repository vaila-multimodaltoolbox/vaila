# Video Compression Tools

## Overview

The Video Compression Tools module provides comprehensive video compression capabilities using multiple codecs (H.264, H.265, H.266) with hardware acceleration support. These tools enable efficient video size reduction while maintaining quality for biomechanical analysis workflows.

## Features

- **Multiple Codec Support**: H.264 (AVC), H.265 (HEVC), and H.266 (VVC) compression
- **Hardware Acceleration**: Automatic detection of NVIDIA NVENC and macOS VideoToolbox
- **Batch Processing**: Compress multiple videos simultaneously
- **Quality Control**: Configurable quality settings with CRF (H.264/H.265) or QP (H.266)
- **Resolution Options**: Keep original or downscale to common resolutions
- **GUI & CLI Interfaces**: Use from the vailá GUI or command line with `argparse`
- **Cross-Platform**: Windows, Linux, and macOS support

## Supported Codecs

### H.264 (AVC) — `compress_videos_h264.py`

- **Compatibility**: Universal support across all devices and platforms
- **Compression**: Good balance between file size and quality
- **Speed**: Fast encoding, especially with hardware acceleration
- **GPU**: NVIDIA NVENC (`h264_nvenc`), macOS VideoToolbox (`h264_videotoolbox`)
- **Default CRF**: 23

### H.265 (HEVC) — `compress_videos_h265.py`

- **Compression**: Up to 50% smaller files than H.264 at same quality
- **Quality**: Maintains high quality at lower bitrates
- **Speed**: Slower encoding but better results
- **GPU**: NVIDIA NVENC (`hevc_nvenc`), macOS VideoToolbox (`hevc_videotoolbox`)
- **Default CRF**: 28

### H.266 (VVC) — `compress_videos_h266.py`

- **Compression**: Best compression ratio (up to 50% better than H.265)
- **Speed**: **Very slow** — CPU-only (no GPU acceleration available)
- **Encoder**: Requires FFmpeg compiled with `libvvenc` (not in standard builds)
- **Default QP**: 32

## Usage

### GUI Mode (from vailá)

Click **Compress** in the vailá toolbox → choose H.264, H.265, or H.266.

```python
# Programmatic GUI launch
from vaila.compress_videos_h264 import compress_videos_h264_gui
from vaila.compress_videos_h265 import compress_videos_h265_gui
from vaila.compress_videos_h266 import compress_videos_h266_gui

compress_videos_h264_gui()  # H.264 compression
compress_videos_h265_gui()  # H.265 compression
compress_videos_h266_gui()  # H.266 compression
```

### CLI Mode

```bash
# H.264 — basic
python -m vaila.compress_videos_h264 --dir /path/to/videos

# H.264 — custom settings with GPU
python -m vaila.compress_videos_h264 --dir /path/to/videos --preset slow --crf 20 --resolution 1920x1080 --gpu

# H.265 — basic
python -m vaila.compress_videos_h265 --dir /path/to/videos

# H.265 — custom settings, no GPU
python -m vaila.compress_videos_h265 --dir /path/to/videos --preset medium --crf 24 --no-gpu

# H.266 — basic (CPU-only, requires libvvenc)
python -m vaila.compress_videos_h266 --dir /path/to/videos

# H.266 — custom quality
python -m vaila.compress_videos_h266 --dir /path/to/videos --preset slow --qp 28 --resolution 1920x1080
```

### CLI Options

#### H.264 and H.265

| Option               | Default                 | Description                                        |
| -------------------- | ----------------------- | -------------------------------------------------- |
| `--dir`              | (required)              | Directory containing videos                        |
| `--preset`           | `medium`                | Encoding preset: `ultrafast` to `veryslow`         |
| `--crf`              | 23 (H.264) / 28 (H.265) | Quality factor (0-51). Lower = better              |
| `--resolution`       | `original`              | Output resolution (e.g. `1920x1080`) or `original` |
| `--gpu` / `--no-gpu` | auto                    | Force GPU or CPU encoding                          |

#### H.266

| Option         | Default    | Description                                        |
| -------------- | ---------- | -------------------------------------------------- |
| `--dir`        | (required) | Directory containing videos                        |
| `--preset`     | `medium`   | Encoding preset: `ultrafast` to `veryslow`         |
| `--qp`         | `32`       | Quantization Parameter (0-51). Lower = better      |
| `--resolution` | `original` | Output resolution (e.g. `1920x1080`) or `original` |

## Hardware Acceleration

### NVIDIA NVENC (Windows/Linux)

- Auto-detected via `nvidia-smi`
- Verified with a test encode before use
- Uses correct modern presets (`p1`-`p7`)
- Falls back to CPU if NVENC test fails

### macOS VideoToolbox

- Auto-detected on macOS
- Uses `h264_videotoolbox` or `hevc_videotoolbox`

### CPU Fallback

- `libx264` for H.264
- `libx265` for H.265
- `libvvenc` for H.266 (always CPU)

## Configuration Parameters

### Encoding Presets

| Preset    | Speed          | Compression | Use Case        |
| --------- | -------------- | ----------- | --------------- |
| ultrafast | Very Fast      | Lower       | Quick previews  |
| veryfast  | Fast           | Good        | General use     |
| fast      | Medium         | Better      | Balanced        |
| medium    | Slow           | Best        | Default         |
| slow      | Very Slow      | Excellent   | Maximum quality |
| veryslow  | Extremely Slow | Maximum     | Best possible   |

### CRF/QP Guidelines

| Range | Quality           | Use Case                     |
| ----- | ----------------- | ---------------------------- |
| 18-22 | Visually lossless | High-quality archival        |
| 23-28 | Good quality      | General analysis             |
| 29-35 | Acceptable        | Preview, storage-constrained |

## Troubleshooting

### NVENC Not Working (GPU detected but encoder fails)

- Your FFmpeg may not be compiled with NVENC support
- System-packaged FFmpeg (apt, brew) often lacks NVENC
- Install FFmpeg with NVENC: `conda install -c conda-forge ffmpeg` or download from https://www.gyan.dev/ffmpeg/builds/

### H.266 "libvvenc not found"

- Standard FFmpeg does NOT include libvvenc
- Download a "full" or "git" build:
  - **Windows**: https://www.gyan.dev/ffmpeg/builds/
  - **Linux/macOS**: https://johnvansickle.com/ffmpeg/
- Verify: `ffmpeg -encoders | grep vvenc`

### Long Encoding Times

- Use faster preset (e.g. `veryfast`)
- Enable GPU acceleration (`--gpu`)
- H.266 is always slow — consider H.265 instead

## Requirements

### Core

- **FFmpeg** installed and in PATH
- **Python 3.x** with Tkinter

### Optional

- **NVIDIA GPU** + NVENC-capable FFmpeg for hardware acceleration
- **FFmpeg with libvvenc** for H.266 support

### FFmpeg Installation

```bash
# Conda (recommended — includes many encoders)
conda install -c conda-forge ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows — download from https://ffmpeg.org/download.html
```

## Integration with vailá

This module integrates with the vailá toolbox via the **Compress** button in the
Video and Image tools section. The button opens a format selection dialog
(H.264 / H.265 / H.266) and launches the corresponding compression module.

---

📅 **Updated:** 18/02/2026
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
