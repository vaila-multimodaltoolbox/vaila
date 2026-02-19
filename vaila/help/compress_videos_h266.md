# compress_videos_h266

## 📋 Module Information

- **Category:** Tools
- **File:** `vaila/compress_videos_h266.py`
- **Version:** 0.0.2
- **GUI Interface:** ✅ Yes
- **CLI Interface:** ✅ Yes

## 📖 Description

Compresses videos in a specified directory to **H.266 (VVC)** format using FFmpeg
with the `libvvenc` encoder. Supports both a **GUI** (Tkinter dialog) and **CLI** (`argparse`) interface.

### ⚠️ Important Notes

- H.266/VVC encoding is **EXTREMELY SLOW** and CPU-intensive
- **No GPU acceleration** available for VVC in common FFmpeg builds
- Requires a **special FFmpeg build** compiled with `libvvenc` support
- Standard system FFmpeg (apt, brew, conda) usually does **NOT** include `libvvenc`

### Key Features

- **Encoder availability check** before starting (graceful error if `libvvenc` missing)
- **QP-based quality control** (Quantization Parameter 0-51)
- **Resolution control**: keep original or downscale to common resolutions
- **Cross-platform**: Windows, Linux, macOS

## 🚀 Usage

### GUI Mode (from vailá)

Select **Compress → H.266 (VVC)** in the vailá toolbox.

### CLI Mode

```bash
# Basic usage (medium preset, QP 32, original resolution)
python -m vaila.compress_videos_h266 --dir /path/to/videos

# Custom quality and resolution
python -m vaila.compress_videos_h266 --dir /path/to/videos --preset slow --qp 28 --resolution 1920x1080
```

### CLI Options

| Option         | Default    | Description                                           |
| -------------- | ---------- | ----------------------------------------------------- |
| `--dir`        | (required) | Directory containing videos                           |
| `--preset`     | `medium`   | Encoding preset: ultrafast → veryslow                 |
| `--qp`         | `32`       | Quantization Parameter (0-51). Lower = better quality |
| `--resolution` | `original` | Output resolution (e.g. `1920x1080`)                  |

### Getting FFmpeg with libvvenc

- **Windows**: Download a "full" build from https://www.gyan.dev/ffmpeg/builds/
- **Linux/macOS**: Download a static "git" build from https://johnvansickle.com/ffmpeg/

Verify with: `ffmpeg -encoders | grep vvenc`

## 🔧 Main Functions

- `check_libvvenc_available` — Check if FFmpeg has libvvenc support
- `find_videos` — Find video files in a directory
- `create_temp_file_with_videos` — Create temp file list for batch processing
- `run_compress_videos_h266` — Core VVC compression logic
- `get_compression_parameters` — GUI parameter dialog
- `compress_videos_h266_gui` — GUI entry point
- `build_parser` — Build argparse CLI parser
- `main` — CLI/GUI entry point

## 📋 Requirements

- **FFmpeg** compiled with `libvvenc` support (version 7.1+ recommended)
- Python 3.x with Tkinter

---

📅 **Updated:** 18/02/2026
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
