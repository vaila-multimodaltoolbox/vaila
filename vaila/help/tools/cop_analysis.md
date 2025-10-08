# cop_analysis

## 📋 Module Information

- **Category:** Tools
- **File:** `vaila/cop_analysis.py`
- **Lines:** 703
- **Size:** 24820 characters
- **Version:** 1.4 Date: 2024-09-12
- **Author:** Prof. Dr. Paulo R. P. Santiago Version: 1.4 Date: 2024-09-12
- **GUI Interface:** ✅ Yes

## 📖 Description


# Plot the CoP pathway with points only (no connecting lines)
ax2.plot(X_n, Y_n, label='CoP Pathway', color='blue', marker='.', markersize=3, linestyle='None')
Module: cop_analysis.py
Description: This module provides a comprehensive set of tools for analyzing Center of Pressure (CoP) data from force plate measurements.
             CoP data is critical in understanding balance and postural control in various fields such as biomechanics, rehabilitation, and sports science.

             The module includes:
             - Functions for data filtering, specifically using a Butterworth filter, to remove noise and smooth the CoP data.
             - GUI components to allow users to select relevant data headers interactively for analysis.
             - Methods to calculate various postural stability metrics such as Mean Square Displacement (MSD), Power Spectral Density (PSD), and sway density.
             - Spectral feature analysis to quantify different frequency components of the CoP ...

## 🔧 Main Functions

**Total functions found:** 10

- `convert_to_cm`
- `read_csv_full`
- `select2headers`
- `plot_final_figure`
- `analyze_data_2d`
- `main`
- `get_csv_headers`
- `on_select`
- `select_all`
- `unselect_all`




---

📅 **Generated automatically on:** 08/10/2025 14:24:24
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
