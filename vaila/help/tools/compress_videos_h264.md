# compress_videos_h264

## 📋 Module Information

- **Category:** Tools
- **File:** `vaila/compress_videos_h264.py`
- **Lines:** 635
- **Size:** 21986 characters


- **GUI Interface:** ✅ Yes

## 📖 Description


vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/paulopreto/vaila-multimodaltoolbox
Please see AUTHORS for contributors.

Licensed under GNU Lesser General Public License v3.0

compress_videos_h264.py

Description:
This script compresses videos in a specified directory to H.264 format using the FFmpeg tool.
It provides a GUI for selecting the directory containing the videos and processes each video,
saving the compressed versions in a subdirectory named 'compressed_h264'.
The script supports GPU acceleration using NVIDIA NVENC if available or falls back to CPU encoding
with libx264.

Usage:
- Run the script to open a GUI, select the directory containing the videos,
  and the compression process will start automatically.

Requirements:
- FFmpeg must be installed and accessible in the system PATH.
- The script is designed to work on Windows, Linux, and macOS.

Dependencies:
- Python 3.x
- Tkinter (included with Python)
- FFmpeg (...

## 🔧 Main Functions

**Total functions found:** 9

- `is_nvidia_gpu_available`
- `find_videos`
- `create_temp_file_with_videos`
- `run_compress_videos_h264`
- `get_compression_parameters`
- `compress_videos_h264_gui`
- `on_ok`
- `on_cancel`
- `show_help`




---

📅 **Generated automatically on:** 08/10/2025 14:24:24
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
