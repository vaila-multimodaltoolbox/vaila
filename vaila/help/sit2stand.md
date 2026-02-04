# sit2stand

## 📋 Module Information

- **Category:** Uncategorized
- **File:** `vaila\sit2stand.py`
- **Lines:** 1727
- **Size:** 61021 characters
- **Version:** 0.0.7
- **Author:** Prof. Paulo Santiago
- **GUI Interface:** ✅ Yes

## 📖 Description


================================================================================
sit2stand.py - Sit to Stand Analysis Module
================================================================================
Author: Prof. Paulo Santiago
Create: 10 October 2025
Update: 03 February 2026
Version: 0.0.7

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
6. C3D File Analysis: Full support for C3D files extraction (Fx, Fy, Fz, Mx, My, Mz)
7. Center of Pressure (CoP): Shimba (1984) calculation for balance and path analysis
8. Interactive Reporting: Plotly-based reports with interactive CoP and Force plots
9. Stability Analysis: Index of stability measuring deviation from horizontal baseline
10. Energy Expenditure Analysis: Calculates mechanical work and metabolic energy based on body weight
11. Advanced Peak Detection: Uses scipy.signal.find_peaks with configurable parameters

Analysis Capabilities:
----------------------
- Sit-to-stand phase detection with configurable thresholds
- Force impulse calculation with filtered data
- Peak force identification and timing analysis
- Movement timing analysis with onset detection
- Balance assessment via CoP Path and Time-Series
- Butterworth low-pass filtering for noise reduction
- Stability index calculation
- Noise and oscillation analysis during standing phase
- Interactive HTML reports with zoom/pan capabilities




---

📅 **Generated automatically on:** 15/10/2025 08:04:44
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
