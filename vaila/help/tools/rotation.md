# rotation

## 📋 Module Information

- **Category:** Tools
- **File:** `vaila/rotation.py`
- **Lines:** 330
- **Size:** 14118 characters
- **Version:** 1.0
- **Author:** Prof. Dr. Paulo R. P. Santiago
- **GUI Interface:** ❌ No

## 📖 Description


================================================================================
Rotation Tools - 3D Rotation and Transformation Toolkit
================================================================================
Author: Prof. Dr. Paulo R. P. Santiago
Date: 2024-07-19
Version: 1.0

Overview:

This Python script provides a comprehensive set of tools for performing 3D rotation and transformation operations using numpy and scipy. It includes methods for creating orthonormal bases from sets of points, calculating rotation matrices, converting rotations to Euler angles and quaternions, and rotating datasets.

Main Features:

    Orthonormal Base Creation:
        - `createortbase`: Generates an orthonormal base from three 3D points, supporting different configurations ('A', 'B', 'C', 'D').
        - `createortbase_4points`: Constructs an orthonormal base using four 3D points, tailored for both trunk and pelvis configurations.

    Rotation Matrix Calculation:
        - `calcmatrot`: C...

## 🔧 Main Functions

**Total functions found:** 6

- `createortbase`
- `createortbase_4points`
- `calcmatrot`
- `rotmat2euler`
- `rotmat2quat`
- `rotdata`




---

📅 **Generated automatically on:** 08/10/2025 14:24:24
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
