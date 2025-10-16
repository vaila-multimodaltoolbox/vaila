# modifylabref

## 📋 Module Information

- **Category:** Processing
- **File:** `vaila\modifylabref.py`
- **Lines:** 491
- **Size:** 17284 characters
- **Version:** 0.0.3
- **Author:** Paulo Roberto Pereira Santiago
- **GUI Interface:** ❌ No

## 📖 Description


Project: vailá Multimodal Toolbox
Script: modifylabref.py - Custom 3D Rotation Processing Toolkit

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 21 Sep 2024
Update Date: 17 Jul 2025
Version: 0.0.3

Description:
    This script is designed for rotating and transforming 3D motion capture data.
    It allows the application of predefined or custom rotation angles (in degrees) and
    rotation orders (e.g., 'xyz', 'zyx') to sets of 3D data points (e.g., from CSV files).

    Main Features:
    1. Predefined Rotations:
        - Options 'A', 'B', and 'C' apply standard rotation transformations:
          'A' applies a 180 degree rotation around the Z-axis.
          'B' applies a 90 degree clockwise rotation around the Z-axis.
          'C' applies a 90 degree counterclockwise rotation around the Z-axis.

    2. Custom Rotation:
        - Users can input custom angles in the format: [x, y, z]
     ...

## 🔧 Main Functions

**Total functions found:** 9

- `rotdata`
- `modify_lab_coords`
- `parse_custom_rotation_input`
- `get_labcoord_angles`
- `detect_column_precision`
- `save_with_original_precision`
- `process_files`
- `run_modify_labref`
- `main`




---

📅 **Generated automatically on:** 15/10/2025 08:04:44
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
