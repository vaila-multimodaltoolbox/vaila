# vaila_lensdistortvideo

## 📋 Module Information

- **Category:** Tools
- **File:** `vaila\vaila_lensdistortvideo.py`
- **Lines:** 349
- **Size:** 11795 characters
- **Version:** 0.1.4
- **Author:** Paulo R. P. Santiago
- **GUI Interface:** ✅ Yes
- **CLI Support:** ✅ Yes

## 📖 Description

===============================================================================
vaila_lensdistortvideo.py
===============================================================================
Author: Paulo R. P. Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 10 October 2024
Update Date: 06 February 2026
Version: 0.1.4
Python Version: 3.12.12
===============================================================================

# Camera Calibration Parameters and Their Meanings

This script processes videos by applying lens distortion correction based on
intrinsic camera parameters and distortion coefficients. It also demonstrates
how to calculate these parameters using field of view (FOV) and resolution.

## Intrinsic Camera Parameters:

1. fx, fy (Focal Length):
   - Represent the focal length of the lens in pixels along the x-axis (fx) and y-axis (fy).
   - Larger values indicate a narrower field of view.
   - Calculated using the formula:
     fx = (width / 2) / tan(horizonta...

## 🔧 Main Functions

**Total functions found:** 5

- `load_distortion_parameters`
- `process_video`
- `select_directory`
- `select_file`
- `run_distortvideo`

## 🚀 Usage

### GUI Mode (Default)

Run the script without arguments to use the graphical interface:

```bash
python vaila_lensdistortvideo.py
```

### CLI Mode (Automation)

Run with command-line arguments for pipeline integration:

```bash
python vaila_lensdistortvideo.py --input_dir "/path/to/videos" --params_file "/path/to/params.csv"
```

Optional: Specify output directory

```bash
python vaila_lensdistortvideo.py --input_dir "..." --params_file "..." --output_dir "/path/to/output"
```

---

📅 **Generated automatically on:** 15/10/2025 08:04:44
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
