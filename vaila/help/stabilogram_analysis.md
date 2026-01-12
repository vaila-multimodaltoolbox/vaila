# stabilogram_analysis

## 📋 Module Information

- **Category:** Tools
- **File:** `vaila\stabilogram_analysis.py`
- **Lines:** 389
- **Size:** 13025 characters
- **Version:** 1.3
- **Author:** Prof. Dr. Paulo R. P. Santiago
- **GUI Interface:** ❌ No

## 📖 Description


Module: stabilogram_analysis.py
Description:
This module provides a comprehensive set of functions for analyzing Center of Pressure (CoP) data obtained from force plate measurements in postural control studies. The functions included in this module allow for the calculation of various stabilometric parameters and the generation of stabilogram plots to evaluate balance and stability.

The main features of this module include:
- **Root Mean Square (RMS) Calculation**: Computes the RMS of the CoP displacement in the mediolateral (ML) and anteroposterior (AP) directions, providing a measure of postural sway.
- **Speed Computation**: Calculates the instantaneous speed of the CoP using a Savitzky-Golay filter to smooth the data, allowing for analysis of the dynamics of postural control.
- **Power Spectral Density (PSD) Analysis**: Computes the PSD of the CoP signals using Welch's method, providing insight into the frequency components of postural sway.
- **Mean Square Displacement (MSD) Cal...

## 🔧 Main Functions

**Total functions found:** 11

- `compute_rms`
- `compute_speed`
- `compute_power_spectrum`
- `compute_msd`
- `count_zero_crossings`
- `count_peaks`
- `compute_sway_density`
- `compute_total_path_length`
- `plot_stabilogram`
- `plot_power_spectrum`
- `save_metrics_to_csv`




---

📅 **Generated automatically on:** 15/10/2025 08:04:44
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
