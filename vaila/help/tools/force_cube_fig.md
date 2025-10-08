# force_cube_fig

## 📋 Module Information

- **Category:** Tools
- **File:** `vaila/force_cube_fig.py`
- **Lines:** 1930
- **Size:** 65744 characters
- **Version:** 0.5
- **Author:** Prof. Dr. Paulo R. P. Santiago Ligia
- **GUI Interface:** ✅ Yes

## 📖 Description


================================================================================
Force Platform Data Analysis Toolkit - force_cube_fig.py
================================================================================
Author: Prof. Dr. Paulo R. P. Santiago Ligia
Date: 9 September 2024
Version: 0.5
Python Version: 3.11

Description:
------------
This script processes biomechanical data from force platforms, analyzing
the vertical ground reaction force (VGRF) to compute key metrics, including:
- Peak forces, time intervals, impulse, rate of force development (RFD),
  and stiffness parameters.
The results are visualized through interactive plots and saved to CSV files for
further analysis. The script allows batch processing of multiple files and provides
descriptive statistics for all analyzed data.

Key Functionalities:
---------------------
1. Data Selection:
   - Allows the user to select input CSV files containing biomechanical data.
   - Prompts the user to specify output directori...

## 🔧 Main Functions

**Total functions found:** 20

- `select_source_directory`
- `select_output_directory`
- `select_body_weight`
- `process_file`
- `batch_process_directory`
- `select_headers_and_load_data`
- `create_main_output_directory`
- `prompt_user_input`
- `butterworthfilt`
- `calculate_median`
- `find_active_indices`
- `build_headers`
- `calculate_loading_rates`
- `logistic_ogive`
- `fit_stiffness_models`
- `calculate_cube_values`
- `makefig1`
- `makefig2`
- `makefig3`
- `makefig4`




---

📅 **Generated automatically on:** 08/10/2025 14:24:24
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
