# readc3d_export

## 📋 Module Information

- **Category:** Processing
- **File:** `vaila/readc3d_export.py`
- **Lines:** 1255
- **Size:** 47131 characters
- **Version:** 25 September 2024
- **Author:** Prof. Paulo R. P. Santiago
- **GUI Interface:** ✅ Yes

## 📖 Description


===============================================================================
readc3d_export.py
===============================================================================
Author: Prof. Paulo R. P. Santiago
Version: 25 September 2024
Update: 15 September 2025
Version updated: 0.0.3
Python Version: 3.12.11

Description:
This script processes .c3d files, extracting marker data, analog data, events, and points residuals,
and saves them into CSV files. It also allows the option to save the data in Excel format.
The script leverages Dask for efficient data handling and processing, particularly useful
when working with large datasets.

Features:
- Extracts and saves marker data with time columns.
- Extracts and saves analog data with time columns, including their units.
- Extracts and saves events with their labels and times.
- Extracts and saves points residuals with time columns.
- Supports saving the data in CSV format.
- Optionally saves the data in Excel format (can be slow for l...

## 🔧 Main Functions

**Total functions found:** 15

- `save_info_file`
- `save_short_info_file`
- `save_events`
- `importc3d`
- `save_empty_file`
- `save_platform_data`
- `save_rotation_data`
- `save_meta_points_data`
- `save_header_summary`
- `save_parameter_groups`
- `save_data_statistics`
- `save_to_files`
- `convert_c3d_to_csv`
- `print_complete_data_structure`
- `batch_convert_c3d_to_csv`




---

📅 **Generated automatically on:** 08/10/2025 14:24:24
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
