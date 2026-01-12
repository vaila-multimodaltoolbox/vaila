# vaila_lensdistortvideo

## 📋 Module Information

- **Category:** Tools
- **File:** `vaila\vaila_lensdistortvideo.py`
- **Lines:** 349
- **Size:** 11795 characters
- **Version:** 0.0.1
- **Author:** Prof. Paulo R. P. Santiago
- **GUI Interface:** ✅ Yes

## 📖 Description


===============================================================================
vaila_lensdistortvideo.py
===============================================================================
Author: Prof. Paulo R. P. Santiago
Date: 20 December 2024
Version: 0.0.1
Python Version: 3.12.8
===============================================================================

Camera Calibration Parameters and Their Meanings
=================================================

This script processes videos by applying lens distortion correction based on
intrinsic camera parameters and distortion coefficients. It also demonstrates
how to calculate these parameters using field of view (FOV) and resolution.

Intrinsic Camera Parameters:
-----------------------------
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




---

📅 **Generated automatically on:** 15/10/2025 08:04:44
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
