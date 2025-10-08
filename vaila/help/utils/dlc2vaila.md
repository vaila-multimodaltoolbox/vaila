# dlc2vaila

## 📋 Module Information

- **Category:** Utils
- **File:** `vaila/dlc2vaila.py`
- **Lines:** 192
- **Size:** 7719 characters
- **Version:** 1.0.0
- **Author:** Prof. Dr. Paulo Santiago
- **GUI Interface:** ✅ Yes

## 📖 Description


Script: dlc2vaila.py
Author: Prof. Dr. Paulo Santiago
Version: 1.0.0
Last Updated: December 9, 2024

Description:
    This script converts DLC (DeepLabCut) CSV files into a format compatible with
    the vailá multimodal toolbox. It processes all CSV files from a specified input
    directory, adjusts their structure, and saves the converted files in a newly
    created directory with a timestamped name.

    The conversion process includes:
    - Retaining only the third header line from the original DLC file.
    - Removing the first column temporarily, processing the remaining data, and re-adding the first column.
    - Excluding every third column from the data.
    - Generating a new header with the format: 'frame, p1_x, p1_y, p2_x, p2_y, ...'.
    - Saving the processed files in a dedicated output directory with a timestamp.

Usage:
    - Run the script to select an input directory containing DLC CSV files.
    - The script will process each file and save the converted outputs i...

## 🔧 Main Functions

**Total functions found:** 2

- `process_csv_files_with_numpy`
- `batch_convert_dlc`




---

📅 **Generated automatically on:** 08/10/2025 14:24:24
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
