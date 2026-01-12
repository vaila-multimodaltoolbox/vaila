# readcsv_export

## 📋 Module Information

- **Category:** Processing
- **File:** `vaila\readcsv_export.py`
- **Lines:** 971
- **Size:** 36912 characters
- **Version:** 25 September 2024
- **Author:** Prof. Paulo R. P. Santiago
- **GUI Interface:** ✅ Yes

## 📖 Description


===============================================================================
readcsv_export.py
===============================================================================
Author: Prof. Paulo R. P. Santiago
Version: 25 September 2024
Update: 04 September 2025
Version updated: 0.1.1
Python Version: 3.12.11

Description:
This script provides functionality to convert CSV files containing point and analog data
into the C3D format, commonly used for motion capture data analysis. The script uses the
ezc3d library to create C3D files from CSV inputs while sanitizing and formatting the data
to ensure compatibility with the C3D standard.

Main Features:
- Reads point and analog data from user-selected CSV files.
- Sanitizes headers to remove unwanted characters and ensure proper naming conventions.
- Handles user input for data rates, unit conversions, and sorting preferences.
- Converts the CSV data into a C3D file with appropriately formatted point and analog data.
- Provides a user in...

## 🔧 Main Functions

**Total functions found:** 8

- `sanitize_header`
- `validate_and_filter_columns`
- `get_conversion_factor`
- `convert_csv_to_c3d`
- `create_c3d_from_csv`
- `batch_convert_csv_to_c3d`
- `auto_create_c3d_from_csv`
- `on_submit`




---

📅 **Generated automatically on:** 15/10/2025 08:04:44
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
