# filter_utils

## 📋 Module Information

- **Category:** Processing
- **File:** `vaila\filter_utils.py`
- **Lines:** 110
- **Size:** 3926 characters
- **Version:** 1.1
- **Author:** Prof. Dr. Paulo R. P. Santiago
- **GUI Interface:** ❌ No

## 📖 Description


Module: filter_utils.py
Description: This module provides a unified and flexible Butterworth filter function for low-pass and band-pass filtering of signals. The function supports edge effect mitigation through optional signal padding and uses second-order sections (SOS) for improved numerical stability.

Author: Prof. Dr. Paulo R. P. Santiago
Version: 1.1
Date: 2024-09-12

Changelog:
- Version 1.1 (2024-09-12):
  - Modified `butter_filter` to handle multidimensional data.
  - Adjusted padding length dynamically based on data length.
  - Fixed issues causing errorrs when data length is less than padding length.

Usage Example:
- Low-pass filter:
  `filtered_data_low = butter_filter(data, fs=1000, filter_type='low', cutoff=10, order=4)`

- Band-pass filter:
  `filtered_data_band = butter_filter(data, fs=1000, filter_type='band', lowcut=5, highcut=15, order=4)`


## 🔧 Main Functions

**Total functions found:** 1

- `butter_filter`




---

📅 **Generated automatically on:** 15/10/2025 08:04:44
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
