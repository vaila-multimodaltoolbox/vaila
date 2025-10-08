# animal_open_field

## 📋 Module Information

- **Category:** Analysis
- **File:** `vaila/animal_open_field.py`
- **Lines:** 1174
- **Size:** 40897 characters
- **Version:** 2.1.0
- **Author:** Prof. Paulo R. P. Santiago
- **GUI Interface:** ✅ Yes

## 📖 Description


===============================================================================
animal_open_field.py
===============================================================================
Author: Prof. Paulo R. P. Santiago
Date: 16 December 2024
Version: 2.1.0
Python Version: 3.11.11

Description:
------------
This script processes movement data of animals in an open field test, performing
comprehensive kinematic analyses and generating visualizations to evaluate animal behavior.

Key Features:
-------------
- Reads movement data from .csv files (X and Y positions over time).
- Calculates key metrics, including:
  - Total distance traveled.
  - Average speed.
  - Time stationary (speed < 0.05 m/s).
  - Time spent in defined speed ranges (0-45 m/min).
- Analyzes the time spent in specific zones of a 60x60 cm open field, divided into
  3x3 grid cells of 20x20 cm each, including:
  - Percentage and count of time in each zone.
  - Percentage and count of time in the center zone and border areas....

## 🔧 Main Functions

**Total functions found:** 20

- `load_and_preprocess_data`
- `adjust_to_bounds`
- `butter_lowpass_filter`
- `define_zones`
- `define_center_zone`
- `calculate_zone_occupancy`
- `calculate_center_and_border_occupancy`
- `calculate_kinematics`
- `plot_pathway`
- `plot_heatmap`
- `plot_center_and_border_heatmap`
- `plot_speed_ranges`
- `plot_speed_over_time_with_tags`
- `save_results_to_csv`
- `save_position_data`
- `process_open_field_data`
- `process_all_files_in_directory`
- `run_animal_open_field`
- `plot_filtering_comparison`
- `save_settings`




---

📅 **Generated automatically on:** 08/10/2025 14:24:24
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
