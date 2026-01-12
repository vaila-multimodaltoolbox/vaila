# sit2stand

## 📋 Module Information

- **Category:** Uncategorized
- **File:** `vaila\sit2stand.py`
- **Lines:** 1727
- **Size:** 61021 characters
- **Version:** 0.0.3
- **Author:** Prof. Paulo Santiago
- **GUI Interface:** ✅ Yes

## 📖 Description


================================================================================
sit2stand.py - Sit to Stand Analysis Module
================================================================================
Author: Prof. Paulo Santiago
Create: 10 October 2025
Update: 14 October 2025
Version: 0.0.3

Description:
------------
This module provides comprehensive functionality for analyzing sit-to-stand movements using force plate data.
It supports batch processing of C3D and CSV files with TOML configuration files and Butterworth filtering.

Key Features:
-------------
1. Batch Processing: Analyze multiple files simultaneously
2. File Format Support: C3D (with ezc3d library) and CSV file formats
3. TOML Configuration: All parameters can be defined in TOML configuration files
4. Butterworth Filtering: Configurable low-pass filtering with user-defined parameters
5. Column Selection: Interactive column selection with detailed file information
6. C3D File Analysis: Full support for C3D files w...

## 🔧 Main Functions

**Total functions found:** 20

- `main`
- `run_cli_mode`
- `get_default_config`
- `run_gui_mode`
- `load_toml_config`
- `find_analysis_files`
- `select_or_confirm_column`
- `auto_detect_force_column`
- `verify_column_exists`
- `butterworth_filter`
- `run_batch_analysis`
- `generate_batch_report`
- `read_c3d_file`
- `print_c3d_info`
- `suggest_force_column`
- `read_csv_file`
- `analyze_sit_to_stand`
- `detect_sit_to_stand_phases`
- `detect_ascending_threshold`
- `find_peaks_in_segment`




---

📅 **Generated automatically on:** 15/10/2025 08:04:44
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
