# readcsv

## 📋 Module Information

- **Category:** Processing
- **File:** `vaila/readcsv.py`
- **Version:** 0.3.122
- **Author:** Paulo Roberto Pereira Santiago
- **Updated:** 06/09/2026
- **GUI Interface:** ✅ Yes

## 📖 Description

Project: vailá Multimodal Toolbox
Script: readcsv.py - Read CSV File

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 29 July 2024
Update Date: 06 September 2026
Version: 0.3.122

Description:
    This script provides tools for reading CSV files and displaying their contents in 3D.
    It includes functions for:
    - Detecting the delimiter used in the file.
    - Detecting if the file has a header.
    - Direct recognition and parsing of calibration model CSV tables (e.g. soccerfield_kiki.csv).
    - Selecting markers to display with select-all preset.
    - Visualizing the data using PyVista, Open3D, or Matplotlib (showc3d).

Usage:
    Run the script from the command line:
        python readcsv.py

Requirements:
    - Python 3.x
    - pandas
    - numpy
    - matplotlib
    - tkinter
    - rich

License:
    This project is licensed under the terms of GNU General Public License v3.0.

Change History:
    - v0.0.3: Added support for CSV, TXT and TSV files, improved UI
    - v0.0.2: Added support for CS...

## 🔧 Main Functions

**Total functions found:** 20

- `show_csv_optimized`
- `show_csv_matplotlib_turbo`
- `headersidx`
- `reshapedata`
- `detect_delimiter`
- `detect_has_header`
- `select_file`
- `choose_visualizer`
- `select_markers_csv`
- `select_headers_gui`
- `get_csv_headers`
- `show_csv_open3d`
- `show_csv_matplotlib`
- `detect_units`
- `ask_user_units`
- `read_csv_generic`
- `show_csv`
- `update`
- `play_pause`
- `on_key`




---

📅 **Generated automatically on:** 15/10/2025 08:04:44
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
