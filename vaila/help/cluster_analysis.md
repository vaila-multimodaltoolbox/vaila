# cluster_analysis

## 📋 Module Information

- **Category:** Analysis
- **File:** `vaila\cluster_analysis.py`
- **Lines:** 515
- **Size:** 18621 characters
- **Version:** 1.0
- **Author:** Prof. Dr. Paulo R. P. Santiago
- **GUI Interface:** ✅ Yes

## 📖 Description


================================================================================
cluster_analysis.py

Cluster Data Analysis Toolkit for Motion Capture
================================================================================
Author: Prof. Dr. Paulo R. P. Santiago
Date: 2024-07-19
Version: 1.0

Overview:

This Python script processes motion capture data for trunk and pelvis rotations using clusters of anatomical markers. It reads CSV files with marker positions, computes orthonormal bases for the clusters, calculates Euler angles, and optionally compares these angles with anatomical reference data. The script generates 3D visualizations of clusters and saves results to CSV for further analysis.

Main Features:

    1. Data Input and Filtering:
        - Reads CSV files containing 3D motion capture data of anatomical markers.
        - Applies a Butterworth filter for noise reduction.

    2. Orthonormal Bases and Euler Angles:
        - Calculates orthonormal bases for the anato...

## 🔧 Main Functions

**Total functions found:** 3

- `save_results_to_csv`
- `read_anatomical_csv`
- `analyze_cluster_data`




---

📅 **Generated automatically on:** 15/10/2025 08:04:44
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
