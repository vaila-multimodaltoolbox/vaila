# imu_analysis

## 📋 Module Information

- **Category:** Analysis
- **File:** `vaila\imu_analysis.py`
- **Lines:** 410
- **Size:** 13033 characters
- **Version:** 1.2
- **Author:** Prof. Ph.D. Paulo Santiago (paulosantiago@usp.br)
- **GUI Interface:** ✅ Yes

## 📖 Description


================================================================================
IMU Analysis Tool - imu_analysis.py
================================================================================
Author: Prof. Ph.D. Paulo Santiago (paulosantiago@usp.br)
Date: 2024-11-21
Version: 1.2

Description:
------------
This script performs IMU sensor data analysis from CSV and C3D files.
It processes accelerometer and gyroscope data, calculates tilt angles and Euler angles,
and generates graphs and CSV files with the processed results. File names for outputs
include the prefix of the processed file.

Features:
- Support for CSV and C3D files.
- Processing of accelerometer and gyroscope data.
- Calculation of tilt angles and Euler angles.
- Saving graphs in PNG format with file-specific prefixes.
- Exporting processed data to uniquely named CSV files.
- Graphical interface for selecting directories and headers.
- Automatic processing of default headers if no selection is made.

Requirements:
-...

## 🔧 Main Functions

**Total functions found:** 6

- `importc3d`
- `imu_orientations`
- `plot_and_save_graphs`
- `save_results_to_csv`
- `plot_and_save_sensor_data`
- `analyze_imu_data`




---

📅 **Generated automatically on:** 15/10/2025 08:04:44
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
