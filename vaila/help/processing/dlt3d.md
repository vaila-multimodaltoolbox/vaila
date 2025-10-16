# dlt3d

## 📋 Module Information

- **Category:** Processing
- **File:** `vaila\dlt3d.py`
- **Lines:** 271
- **Size:** 10271 characters
- **Version:** 0.0.3
- **Author:** Paulo Roberto Pereira Santiago
- **GUI Interface:** ✅ Yes

## 📖 Description


================================================================================
Script: dlt3d.py
================================================================================
vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

================================================================================
Author: Paulo Roberto Pereira Santiago
Version: 0.0.3
Create: 24 February, 2025
Last Updated: 02 August, 2025

Description:
    This script calculates the Direct Linear Transformation (DLT) parameters for 3D coordinate transformations.
    It uses pixel coordinates from video calibration data and corresponding real-world 3D coordinates to compute the 11
    DLT parameters for each frame (or uses a single row of real-world coordinates for all frames).

    New Features:
      - Generates a REF3D template (with _x, _y, _z columns) from the pixel file.
      - ...

## 🔧 Main Functions

**Total functions found:** 6

- `read_pixel_file`
- `read_ref3d_file`
- `calculate_dlt3d_params`
- `process_files`
- `save_dlt_parameters`
- `main`




---

📅 **Generated automatically on:** 15/10/2025 08:04:44
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
