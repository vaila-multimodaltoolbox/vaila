# compress_videos_h265

## 📋 Module Information

- **Category:** Tools
- **File:** `vaila\compress_videos_h265.py`
- **Lines:** 657
- **Size:** 23276 characters


- **GUI Interface:** ✅ Yes

## 📖 Description


vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/paulopreto/vaila-multimodaltoolbox
Please see AUTHORS for contributors.

Licensed under GNU Lesser General Public License v3.0

compress_videos_h265.py

Description:
This script compresses videos in a specified directory to H.265/HEVC format using the FFmpeg tool.
It provides a GUI for selecting the directory containing the videos, and processes each video,
saving the compressed versions in a subdirectory named 'compressed_h265'.
The script supports GPU acceleration using NVIDIA NVENC if available, or falls back to CPU encoding
with libx265.

The script has been updated to work on Windows, Linux, and macOS.
It includes cross-platform detection of NVIDIA GPUs to utilize GPU acceleration where possible.
On systems without an NVIDIA GPU (e.g., macOS), the script defaults to CPU-based compression.

Usage:
- Run the script to open a GUI, select the directory containing the videos, and...

## 🔧 Main Functions

**Total functions found:** 9

- `is_nvidia_gpu_available`
- `find_videos`
- `create_temp_file_with_videos`
- `run_compress_videos_h265`
- `get_compression_parameters`
- `compress_videos_h265_gui`
- `on_ok`
- `on_cancel`
- `show_help`




---

📅 **Generated automatically on:** 15/10/2025 08:04:44
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
