# compress_videos_h264

## 📋 Module Information

- **Category:** Tools
- **File:** `vaila/compress_videos_h264.py`
- **Version:** 0.3.24
- **GUI Interface:** ✅ Yes
- **CLI Interface:** ✅ Yes

## 📖 Description

Compresses videos in a specified directory to **H.264 (AVC)** format using FFmpeg.
Supports both a **GUI** (Tkinter dialog) and **CLI** (`argparse`) interface.

### Key Features

- **GPU acceleration** with NVIDIA NVENC (auto-detected via `nvidia-smi`)
- **Parallel processing**: speed up batch compression with multiple workers
- **Adaptive compression**: automatically discard output files larger than input
- **macOS VideoToolbox** hardware encoding support
- **CPU fallback** to `libx264` when no GPU is available
- **Resolution control**: keep original or downscale to common resolutions
- **Preset and CRF** selection for quality/speed trade-off
- **Cross-platform**: Windows, Linux, macOS

## 🚀 Usage

### GUI Mode (from vailá)

Select **Compress → H.264 (AVC)** in the vailá toolbox.

### CLI Mode

```bash
# Basic usage (medium preset, CRF 23, original resolution)
python -m vaila.compress_videos_h264 --dir /path/to/videos

# With GPU acceleration
python -m vaila.compress_videos_h264 --dir /path/to/videos --gpu

# Custom quality and resolution
python -m vaila.compress_videos_h264 --dir /path/to/videos --preset slow --crf 20 --resolution 1920x1080

# Force CPU only
python -m vaila.compress_videos_h264 --dir /path/to/videos --no-gpu
```

### CLI Options

| Option               | Default    | Description                            |
| -------------------- | ---------- | -------------------------------------- |
| `--dir`              | (required) | Directory containing videos            |
| `--preset`           | `medium`   | Encoding preset: ultrafast → veryslow  |
| `--crf`              | `23`       | Quality (0-51). Lower = better quality |
| `--resolution`       | `original` | Output resolution (e.g. `1920x1080`)   |
| `--gpu` / `--no-gpu` | auto       | Force GPU or CPU encoding              |
| `--workers` / `-w`   | `1`        | Number of parallel workers             |

## 🔧 Main Functions

- `is_nvidia_gpu_available` — Detect NVIDIA GPU via `nvidia-smi`
- `verify_nvenc_encoder` — Test that h264_nvenc actually works
- `find_videos` — Find video files in a directory
- `create_temp_file_with_videos` — Create temp file list for batch processing
- `run_compress_videos_h264` — Core compression logic
- `get_compression_parameters` — GUI parameter dialog
- `compress_videos_h264_gui` — GUI entry point
- `build_parser` — Build argparse CLI parser
- `main` — CLI/GUI entry point

## 📋 Requirements

- **FFmpeg** installed and in PATH
- Python 3.x with Tkinter
- Optional: NVIDIA GPU with NVENC-capable FFmpeg

---

📅 **Updated:** 05/03/2026
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
