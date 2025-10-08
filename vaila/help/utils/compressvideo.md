# compressvideo

## 📋 Module Information

- **Category:** Utils
- **File:** `vaila/compressvideo.py`
- **Lines:** 181
- **Size:** 5462 characters


- **GUI Interface:** ✅ Yes

## 📖 Description


# vailá - Multimodal Toolbox
# © Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
# https://github.com/paulopreto/vaila-multimodaltoolbox
# Please see AUTHORS for contributors.
#
# Licensed under GNU Lesser General Public License v3.0
#
# compress_videos.py
# This script compresses videos in a specified directory to either H.264 or H.265/HEVC format
# using the FFmpeg tool. It provides a GUI for selecting the directory containing the videos,
# allows the user to choose the desired codec, and then processes each video, saving the
# compressed versions in a subdirectory named 'compressed_[codec]'.
#
# Usage:
# Run the script to open the GUI, select the directory containing videos, choose the codec,
# and the compression process will start automatically.
#
# Requirements:
# - FFmpeg must be installed and accessible in the system PATH.
# - This script is designed to work in a Conda environment where FFmpeg is
#   installed via conda-forge.
#
# Dependencies:
# - Python 3.12.9
# -...

## 🔧 Main Functions

**Total functions found:** 5

- `check_ffmpeg_encoder`
- `run_compress_videos`
- `ask_codec_selection`
- `compress_videos_gui`
- `on_ok`




---

📅 **Generated automatically on:** 08/10/2025 14:24:24
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
