# cube2d_kinematics

## 📋 Module Information

- **Category:** Analysis
- **File:** `vaila\cube2d_kinematics.py`
- **Lines:** 722
- **Size:** 23935 characters
- **Version:** 0.0.3
- **Author:** Prof. Dr. Paulo Roberto Pereira Santiago
- **GUI Interface:** ✅ Yes

## 📖 Description


===============================================================================
cube2d_kinematics.py
===============================================================================
Author: Prof. Dr. Paulo Roberto Pereira Santiago
created: 2025-01-16
updated: 2025-06-09
version: 0.0.3
python version: 3.12.9

Description:
This module provides functionality for analyzing 2D kinematics data from cube-based movement assessments.

Key Features:
- Processes CSV files containing 2D position data (x,y coordinates)
- Calculates kinematic metrics including:
  - Total distance traveled
  - Average speed
  - Time spent stationary
  - Total movement time
- Divides movement space into 9 quadrants for spatial analysis
- Generates visualizations:
  - Movement pathway plots with quadrant overlays
  - Color-coded speed profiles
- Handles batch processing of multiple files
- Outputs results to organized directories with timestamps
- Provides GUI interface for:
  - File/directory selection
  - Parameter i...

## 🔧 Main Functions

**Total functions found:** 10

- `show_instructions`
- `load_quadrants`
- `load_data`
- `butter_lowpass_filter`
- `calculate_distance`
- `calculate_speed`
- `plot_pathway_with_quadrants`
- `process_file`
- `run_cube2d_kinematics`
- `on_continue`




---

📅 **Generated automatically on:** 15/10/2025 08:04:44
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
