# vaila_datdistort

## 📋 Module Information

- **Category:** Tools
- **File:** `vaila\vaila_datdistort.py`
- **Lines:** 269
- **Size:** 9032 characters
- **Version:** 0.0.4
- **Author:** Prof. Dr. Paulo R. P. Santiago
- **GUI Interface:** ✅ Yes
- **CLI Support:** ✅ Yes

## 📖 Description

================================================================================
vaila_datdistort.py
================================================================================
vailá - Multimodal Toolbox
Author: Prof. Dr. Paulo R. P. Santiago
https://github.com/paulopreto/vaila-multimodaltoolbox
Date: 03 April 2025
Update: 06 February 2026
Version: 0.0.4
Python Version: 3.12.12

## Description:

This tool applies lens distortion correction to 2D coordinates from a DAT file
using the same camera calibration parameters as vaila_lensdistortvideo.py.

## New Features in This Version:

1. CLI support for pipeline integration.
2. Fixed issue with column order in output file.
3. Improved error handling.

## How to use:

1. Select the distortion parameters CSV file.
2. Select the directory containing CSV/DAT files to process.
3. The script will process all CSV and DAT files in the directory and save the
   results in the output directory.

python vai...

## 🔧 Main Functions

**Total functions found:** 6

- `load_distortion_parameters`
- `undistort_points`
- `process_dat_file`
- `select_file`
- `select_directory`
- `run_datdistort`

## 🚀 Usage

### GUI Mode (Default)

Run the script without arguments to use the graphical interface:

```bash
python vaila_datdistort.py
```

### CLI Mode (Automation)

Run with command-line arguments for pipeline integration:

```bash
python vaila_datdistort.py --input_dir "/path/to/data" --params_file "/path/to/params.csv"
```

Optional: Specify output directory

```bash
python vaila_datdistort.py --input_dir "..." --params_file "..." --output_dir "/path/to/output"
```

---

📅 **Generated automatically on:** 15/10/2025 08:04:44
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
