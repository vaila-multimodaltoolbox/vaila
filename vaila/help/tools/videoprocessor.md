# videoprocessor

## 📋 Module Information

- **Category:** Tools
- **File:** `vaila/videoprocessor.py`
- **Lines:** 788
- **Size:** 29552 characters
- **Version:** merged.

- **GUI Interface:** ✅ Yes

## 📖 Description


videoprocessor.py
vailá - Multimodal Toolbox
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.
Created by Paulo Santiago
Date: 03 April 2025
Updated: 25 July 2025

Licensed under GNU Lesser General Public License v3.0

Description:
This script allows users to process and edit video files, enabling batch processing of videos. Users can choose between two main operations:
1. Merging a video with its reversed version, resulting in a video with double the frames.
2. Splitting each video into two halves and saving only the second half.
The script supports custom text files for batch processing and includes a GUI for directory and file selection.

Key Features:
- Graphical User Interface (GUI) for easy selection of directories and file inputs.
- Batch processing using a text file (`videos_e_frames.txt`) with custom instructions for specifying which videos to process.
- If no text file is provided, the script processes all videos in the source directory.
-...

## 🔧 Main Functions

**Total functions found:** 6

- `check_ffmpeg_installed`
- `detect_hardware_encoder`
- `check_video_size`
- `process_videos_merge`
- `process_videos_split`
- `process_videos_gui`




---

📅 **Generated automatically on:** 08/10/2025 14:24:24
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
