# showc3d

## 📋 Module Information

- **Category:** Visualization
- **File:** `vaila/showc3d.py`
- **Version:** 0.3.122
- **Author:** Prof. Paulo Roberto Pereira Santiago
- **Updated:** 06/09/2026
- **GUI Interface:** ✅ Yes

## 📖 Description

Script: showc3d.py
Author: Prof. Paulo Roberto Pereira Santiago
Date: 29/07/2024
Updated: 06/09/2026
Version: 0.3.122

Description:
------------
This script visualizes marker and calibration keypoint data from C3D files
or direct numpy arrays using Matplotlib in 3D.

Features:
- Automatic unit detection (meters vs millimeters).
- Adapts spatial environment for soccer field / sports court scale
  (pitch boundary lines, halfway line, center circle, ground plane).
- 3D marker text labels with interactive toggle button.
- Frame slider and Play/Pause animation for multi-frame MoCap,
  with streamlined display for single-frame calibration models.
- Standalone CLI execution, GUI invocation, and programmatic `show_points_3d()`.

Usage:
------
1. From GUI: Frame C → Tools → Choose C3D viewer → Matplotlib viewer (showc3d)
2. Command line: `uv run vaila/showc3d.py [path/to/model.c3d]`
3. Programmatic:
   ```python
   from vaila.showc3d import show_c3d, show_points_3d
   show_c3d("soccerfield_kiki.c3d")
   ```

## 🔧 Main Functions

- `load_c3d_file` — Load marker data, header frame rate, and labels from C3D with unit detection.
- `select_markers` — Tkinter dialog to select markers to display.
- `draw_cartesian_axes` — Draw RGB Cartesian axes scaled to the dataset span.
- `draw_soccer_field_features` — Draw pitch boundaries, halfway line, and center circle on Z=0.
- `show_points_3d` — Core 3D scatter and animation viewer for marker and calibration data.
- `main` — CLI entry point.
- `show_c3d` — Convenience launcher alias.




---

📅 **Generated automatically on:** 15/10/2025 08:04:44
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
