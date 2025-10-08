# rm_duplicateframes

## 📋 Module Information

- **Category:** Tools
- **File:** `vaila/rm_duplicateframes.py`
- **Lines:** 437
- **Size:** 15976 characters
- **Version:** 0.0.1
- **Author:** Prof. Dr. Paulo R. P. Santiago
- **GUI Interface:** ✅ Yes

## 📖 Description


================================================================================
Remove Duplicate Frames Tool - rm_duplicateframes.py
================================================================================
Author: Prof. Dr. Paulo R. P. Santiago
Laboratory of Biomechanics and Motor Control (LaBioCoM)
School of Physical Education and Sport of Ribeirão Preto
University of São Paulo (USP)

Contact: paulosantiago@usp.br
Laboratory website: https://github.com/vaila-multimodaltoolbox/vaila

Created: March 19, 2025
Last Updated: March 19, 2025
Version: 0.0.1

Description:
------------
This script removes frames based on a specified pattern (e.g., every 6th frame)
from a sequence of PNG images. It creates a backup of the removed frames and
can regenerate a video with the remaining frames at a specified FPS.

Dependencies:
------------
- Python 3.x
- Tkinter
- FFmpeg (for video creation)

Usage:
------
Run the script and select the directory containing PNG frame sequences.
Specify the ...

## 🔧 Main Functions

**Total functions found:** 9

- `run_rm_duplicateframes`
- `extract_frame_number`
- `remove_frames_by_pattern`
- `update_video_info`
- `create_video`
- `restore_duplicate_frames`
- `copy_backup_to_video_dir`
- `run_rm_duplicateframes`
- `show_completion_message`




---

📅 **Generated automatically on:** 08/10/2025 14:24:24
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
