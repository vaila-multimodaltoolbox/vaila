# interp_smooth_split

## 📋 Module Information

- **Category:** Processing
- **File:** `vaila\interp_smooth_split.py`
- **Lines:** 3283
- **Size:** 131102 characters
- **Version:** 0.0.7
- **Author:** Paulo R. P. Santiago
- **GUI Interface:** ✅ Yes

## 📖 Description


===============================================================================
interp_smooth_split.py
===============================================================================
Author: Paulo R. P. Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 14 October 2024
Update Date: 16 September 2025
Version: 0.0.7
Python Version: 3.12.9

Description:
------------
This script provides functionality to fill missing data in CSV files using
linear interpolation, Kalman filter, Savitzky-Golay filter, nearest value fill,
or to split data into a separate CSV file. It is intended for use in biomechanical
data analysis, where gaps in time-series data can be filled and datasets can be
split for further analysis.

Key Features:
-------------
1. **Data Splitting**:
   - Splits CSV files into two halves for easier data management and analysis.
2. **Padding**:
   - Pads the data with the last valid value to avoid edge effects.

    padding_l...

## 🔧 Main Functions

**Total functions found:** 20

- `save_config_to_toml`
- `load_config_from_toml`
- `generate_report`
- `detect_float_format`
- `savgol_smooth`
- `lowess_smooth`
- `spline_smooth`
- `kalman_smooth`
- `arima_smooth`
- `process_file`
- `run_fill_split_dialog`
- `setup_variables`
- `center_window`
- `on_window_resize`
- `create_dialog_content`
- `create_interpolation_section`
- `create_smoothing_section`
- `create_split_section`
- `create_parameters_section`
- `create_padding_section`




---

📅 **Generated automatically on:** 15/10/2025 08:04:44
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
