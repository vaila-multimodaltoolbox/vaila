# numstepsmp

## 📋 Module Information

- **Category:** Tools
- **File:** `vaila\numstepsmp.py`
- **Lines:** 1473
- **Size:** 55909 characters

- **Author:** Paulo Roberto Pereira Santiago
- **GUI Interface:** ✅ Yes

## 📖 Description


numstepsmp.py

Description:
    Opens a dialog to select a CSV file of foot coordinates
    and calculates the number of steps based on foot position using
    MediaPipe data. Includes Butterworth filtering and advanced metrics using
    multiple markers (ankle, heel, toe).

Author:
    Paulo Roberto Pereira Santiago

Created:
    14 May 2025
Updated:
    16 May 2025

Usage:
    python numstepsmp.py

Dependencies:
    - pandas
    - numpy
    - scipy
    - tkinter (GUI for file selection)
    - matplotlib (optional, for visualization)


## 🔧 Main Functions

**Total functions found:** 16

- `butterworth_filter`
- `filter_signals`
- `calculate_feet_metrics`
- `count_steps_original`
- `count_steps_basic`
- `count_steps_velocity`
- `count_steps_sliding_window`
- `count_steps_mean_y`
- `count_steps_z_depth`
- `detect_foot_strikes_heel_z`
- `detect_foot_strikes`
- `count_steps`
- `export_results`
- `run_numsteps`
- `extract_gait_features`
- `run_numsteps_gui`




---

📅 **Generated automatically on:** 15/10/2025 08:04:44
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
