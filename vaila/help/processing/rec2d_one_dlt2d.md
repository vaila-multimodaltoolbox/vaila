# rec2d_one_dlt2d

## 📋 Module Information

- **Category:** Processing
- **File:** `vaila/rec2d_one_dlt2d.py`
- **Lines:** 211
- **Size:** 7327 characters
- **Version:** 0.0.3
- **Author:** Paulo Santiago
- **GUI Interface:** ✅ Yes

## 📖 Description


================================================================================
Script: rec2d_one_dlt2d.py
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/paulopreto/vaila-multimodaltoolbox
Please see AUTHORS for contributors.

================================================================================
Author: Paulo Santiago
Version: 0.0.3
Created: August 9, 2024
Last Updated: August 02, 2025

Description:
    Optimized batch processing of 2D coordinates reconstruction using corresponding DLT parameters for each frame.
    Processes multiple CSV files containing pixel coordinates and reconstructs them to 2D real-world coordinates.
    The pixel files are expected to use vailá's standard header:
      frame,p1_x,p1_y,p2_x,p2_y,...,pN_x,pN_y
    Uses DLT2D parameters that can vary per frame.

    Optimizations:
    - Pre-allocated NumPy array...

## 🔧 Main Functions

**Total functions found:** 4

- `read_coordinates`
- `rec2d`
- `process_files_in_directory`
- `run_rec2d_one_dlt2d`




---

📅 **Generated automatically on:** 08/10/2025 14:24:24
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
