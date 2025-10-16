# cutvideo

## 📋 Module Information

- **Category:** Tools
- **File:** `vaila\cutvideo.py`
- **Lines:** 925
- **Size:** 33593 characters
- **Version:** 0.0.8
- **Author:** Paulo Roberto Pereira Santiago
- **GUI Interface:** ✅ Yes

## 📖 Description


Project: vailá Multimodal Toolbox
Script: cutvideo.py - Cut Video

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 29 July 2024
Update Date: 28 July 2025
Version: 0.0.8

Description:
This script performs batch processing of videos for cutting videos.


Features:
- Added support for multiple videos.
- Added support for multiple output directories.
- Added support for sync files.

Usage:
- Run the script to open a graphical interface for selecting the input directory
  containing video files (.mp4, .avi, .mov), the output directory, and for
  specifying the MediaPipe configuration parameters.
- Choose whether to enable video resize for better pose detection.
- The script processes each video, generating an output video with overlaid pose
  landmarks, and CSV files containing both normalized and pixel-based landmark
  coordinates in original video dimensions.

Requirements:
- Python 3.12.11
- OpenC...

## 🔧 Main Functions

**Total functions found:** 16

- `save_cuts_to_txt`
- `load_sync_file`
- `load_cuts_from_txt`
- `load_cuts_or_sync`
- `load_sync_file_from_dialog`
- `batch_process_sync_videos`
- `play_video_with_cuts`
- `get_video_path`
- `cleanup_resources`
- `run_cutvideo`
- `draw_controls`
- `show_help_dialog`
- `save_and_generate_videos`
- `batch_process_videos`
- `save_cuts`
- `process_next_video`




---

📅 **Generated automatically on:** 15/10/2025 08:04:44
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
