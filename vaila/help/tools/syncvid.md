# syncvid

## 📋 Module Information

- **Category:** Tools
- **File:** `vaila/syncvid.py`
- **Lines:** 314
- **Size:** 10276 characters
- **Version:** 0.0.2
- **Author:** Paulo Roberto Pereira Santiago
- **GUI Interface:** ✅ Yes

## 📖 Description


Project: vailá Multimodal Toolbox
Script: syncvid.py - Sync Video

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 29 July 2024
Update Date: 28 July 2025
Version: 0.0.2

Description:
This script performs batch processing of videos for sync videos.


Features:
- Added support for sync files.

Usage:
- Run the script to open a graphical interface for selecting the input directory
  containing video files (.mp4, .avi, .mov), the output directory, and for
  specifying the sync file.

Requirements:
- Python 3.12.11
- Tkinter (usually included with Python installations)

Output:
The following files are generated for each processed video:
1. Sync File (`*_sync.txt`):
   The sync file with the sync data.

Example:
- Video: 1.mp4
- Sync: 1_sync.txt
- Output: 1_sync.mp4

How to run:
python syncvid.py

License:
    This project is licensed under the terms of GNU General Public License v3.0.


## 🔧 Main Functions

**Total functions found:** 10

- `get_video_files`
- `write_sync_file`
- `get_sync_info`
- `sync_videos`
- `create_widgets`
- `on_next`
- `select_main_camera`
- `set_main_camera`
- `get_sync_data`
- `on_ok`




---

📅 **Generated automatically on:** 08/10/2025 14:24:24
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
