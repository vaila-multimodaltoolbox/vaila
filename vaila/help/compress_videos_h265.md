# compress_videos_h265

## 📋 Module Information

- **Category:** Tools
- **File:** `vaila/compress_videos_h265.py`
- **Version:** 0.1.2
- **GUI Interface:** ✅ Yes
- **CLI Interface:** ✅ Yes

## 📖 Description

Compresses videos in a specified directory to **H.265 (HEVC)** format using FFmpeg.
Supports both a **GUI** (Tkinter dialog) and **CLI** (`argparse`) interface.

### Key Features

- **GPU acceleration** with NVIDIA NVENC (`hevc_nvenc`, auto-detected via `nvidia-smi`)
- **macOS VideoToolbox** hardware encoding support (`hevc_videotoolbox`)
- **CPU fallback** to `libx265` when no GPU is available
- **Better compression** than H.264 — up to 50% smaller files at same quality
- **Resolution control**: keep original or downscale to common resolutions
- **Recursive search**: process all videos in subdirectories (CLI `--recursive`, GUI subdir depth)
- **Subdir depth**: limit how many levels of subdirectories to include (0 = only selected folder, -1 = unlimited)
- **Cross-platform**: Windows, Linux, macOS

## 🚀 Usage

### GUI Mode (from vailá)

Select **Compress → H.265 (HEVC)** in the vailá toolbox.

### CLI Mode

```bash
# Basic usage (medium preset, CRF 28, original resolution)
python -m vaila.compress_videos_h265 --dir /path/to/videos

# With GPU acceleration
python -m vaila.compress_videos_h265 --dir /path/to/videos --gpu

# Custom quality and resolution
python -m vaila.compress_videos_h265 --dir /path/to/videos --preset slow --crf 24 --resolution 1920x1080

# Recursive with GPU
python -m vaila.compress_videos_h265 --dir /path/to/videos --recursive --gpu

# Recursive with max depth 2
python -m vaila.compress_videos_h265 --dir /path/to/videos -r --depth 2 --gpu

# Force CPU only
python -m vaila.compress_videos_h265 --dir /path/to/videos --no-gpu
```

### CLI Options

| Option               | Default    | Description                                                            |
| -------------------- | ---------- | ---------------------------------------------------------------------- |
| `--dir`              | (required) | Directory containing videos                                            |
| `--recursive` / `-r` | off        | Recurse into subdirectories (output mirrors folder structure)          |
| `--depth` / `-d`     | `-1`       | Max subdir depth when recursive (0=root only, 1/2/3=levels, -1=unlimited) |
| `--preset`           | `medium`   | Encoding preset: ultrafast → veryslow                                  |
| `--crf`              | `28`       | Quality (0-51). Lower = better quality                                  |
| `--resolution`       | `original` | Output resolution (e.g. `1920x1080`)                                  |
| `--gpu` / `--no-gpu` | auto       | Force GPU or CPU encoding                                              |
| `--workers` / `-w`   | `1`        | Number of parallel workers                                             |

## 🔧 Main Functions

- `is_nvidia_gpu_available` — Detect NVIDIA GPU via `nvidia-smi`
- `verify_nvenc_encoder` — Test that hevc_nvenc actually works
- `find_videos` — Find video files in a directory (non-recursive)
- `find_videos_recursive` — Find video files in directory and subdirs (optional max_depth)
- `create_temp_file_with_videos` — Create temp file list for batch processing
- `run_compress_videos_h265` — Core compression logic (accepts video_list of path/output_path pairs)
- `get_compression_parameters` — GUI parameter dialog
- `compress_videos_h265_gui` — GUI entry point
- `build_parser` — Build argparse CLI parser
- `main` — CLI/GUI entry point

## 📋 Requirements

- **FFmpeg** installed and in PATH (with `libx265` support)
- Python 3.x with Tkinter
- Optional: NVIDIA GPU with NVENC-capable FFmpeg

---

📅 **Updated:** 06/03/2026
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
