# rearrange_data

## 📋 Module Information

- **Category:** Processing
- **File:** `vaila/rearrange_data.py`
- **Lines:** 1772
- **Size:** 68536 characters
- **Version:** 0.0.6
- **Author:** Paulo Roberto Pereira Santiago
- **GUI Interface:** ✅ Yes

## 📖 Description


Project: vailá Multimodal Toolbox
Script: rearrange_data.py - CSV Data Rearrangement and Processing Tool

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 08 Oct 2024
Update Date: 25 Jul 2025
Version: 0.0.6

Description:
    This script provides tools for rearranging and processing CSV data files.
    It includes functions for:
    - Reordering columns.
    - Merging and stacking CSV files.
    - Converting MediaPipe data to a format compatible with 'getpixelvideo.py'.
    - Detecting precision and scientific notation in the data.
    - Converting units between various metric systems.
    - Modifying lab reference systems.
    - Saving the second half of each CSV file.

Usage:
    Run the script from the command line:
        python rearrange_data.py

Requirements:
    - Python 3.x
    - pandas
    - numpy
    - tkinter

License:
    This project is licensed under the terms of GNU General Public ...

## 🔧 Main Functions

**Total functions found:** 20

- `detect_column_precision_detailed`
- `save_dataframe_with_precision`
- `detect_precision_and_notation`
- `save_dataframe`
- `get_headers`
- `reshapedata`
- `convert_mediapipe_to_pixel_format`
- `batch_convert_mediapipe`
- `convert_kinovea_to_vaila`
- `batch_convert_kinovea`
- `convert_dvideo_to_vaila`
- `batch_convert_dvideo`
- `convert_yolo_tracker_to_pixel_format`
- `batch_convert_yolo_tracker`
- `rearrange_data_in_directory`
- `setup_empty_gui`
- `setup_large_file_gui`
- `setup_normal_gui`
- `setup_gui`
- `setup_bindings`




---

📅 **Generated automatically on:** 08/10/2025 14:24:24
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
