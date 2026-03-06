# compress_videos_h266

## 📋 Module Information

- **Category:** Tools
- **File:** `vaila/compress_videos_h266.py`
- **Version:** 0.1.2
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

- **Parallel processing**: speed up batch compression with multiple workers
- **Adaptive compression**: automatically discard output files larger than input
- **Encoder availability check** before starting (graceful error if `libvvenc` missing)
- **QP-based quality control** (Quantization Parameter 0-63)
- **Resolution control**: keep original or downscale to common resolutions
- **Recursive search**: process all videos in subdirectories (CLI `--recursive`, GUI subdir depth)
- **Subdir depth**: limit how many levels of subdirectories to include (0 = only selected folder, -1 = unlimited)
- **Cross-platform**: Windows, Linux, macOS

## 🚀 Usage

### GUI Mode (from vailá)

Select **Compress → H.266 (VVC)** in the vailá toolbox.

### CLI Mode

```bash
# Basic usage (medium preset, QP 32, original resolution)
python -m vaila.compress_videos_h266 --dir /path/to/videos

# Recursive (all subdirs)
python -m vaila.compress_videos_h266 --dir /path/to/videos --recursive

# Recursive with max depth 2
python -m vaila.compress_videos_h266 --dir /path/to/videos -r --depth 2

# Custom quality and resolution
python -m vaila.compress_videos_h266 --dir /path/to/videos --preset slow --qp 28 --resolution 1920x1080
```

### CLI Options

| Option               | Default    | Description                                                            |
| -------------------- | ---------- | ---------------------------------------------------------------------- |
| `--dir`              | (required) | Directory containing videos                                            |
| `--recursive` / `-r` | off        | Recurse into subdirectories (output mirrors folder structure)          |
| `--depth` / `-d`     | `-1`       | Max subdir depth when recursive (0=root only, 1/2/3=levels, -1=unlimited) |
| `--preset`           | `medium`   | Encoding preset: ultrafast → veryslow                                  |
| `--qp`               | `32`       | Quantization Parameter (0-63). Lower = better quality                 |
| `--resolution`       | `original` | Output resolution (e.g. `1920x1080`)                                   |
| `--workers` / `-w`   | `1`        | Number of parallel workers                                             |

### Getting FFmpeg with libvvenc

- **Windows**: Download a "full" build from https://www.gyan.dev/ffmpeg/builds/
- **Linux/macOS**: Download a static "git" build from https://johnvansickle.com/ffmpeg/

Verify with: `ffmpeg -encoders | grep vvenc`

## 🔧 Main Functions

- `check_libvvenc_available` — Check if FFmpeg has libvvenc support
- `find_videos` — Find video files in a directory (non-recursive)
- `find_videos_recursive` — Find video files in directory and subdirs (optional max_depth)
- `create_temp_file_with_videos` — Create temp file list for batch processing
- `run_compress_videos_h266` — Core VVC compression logic (accepts video_list of path/output_path pairs)
- `get_compression_parameters` — GUI parameter dialog
- `compress_videos_h266_gui` — GUI entry point
- `build_parser` — Build argparse CLI parser
- `main` — CLI/GUI entry point

## 📋 Requirements

- **FFmpeg** compiled with `libvvenc` support (version 7.1+ recommended)
- Python 3.x with Tkinter

---

📅 **Updated:** 06/03/2026
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
