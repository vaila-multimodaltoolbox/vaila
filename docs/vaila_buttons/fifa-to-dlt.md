# FIFA cams→DLT Button

The **FIFA cams→DLT** button converts camera calibration files from the FIFA official dataset into vailá-compatible DLT files.

## Overview

In the FIFA Skeletal Tracking challenge, camera parameters (K, R, t) are often provided in `.npz` format. This tool converts these parameters into per-frame `.dlt2d` and `.dlt3d` files, allowing vailá's reconstruction modules (`rec2d.py`, `rec3d.py`) to process broadcast footage with moving cameras.

## Key Features

- **Batch Conversion:** Processes entire directories of FIFA `.npz` camera files.
- **Per-Frame Calibration:** Generates one DLT row per video frame, accounting for camera movement (pan, tilt, zoom).
- **Dual Output:** Supports both 2D (field plane) and 3D (world space) DLT parameters.

## Usage

1. Click **FIFA cams→DLT** in Frame B.
2. Select the directory containing the FIFA `cameras/*.npz` files.
3. Choose the output directory for the `.dlt2d` or `.dlt3d` files.
4. The tool will process each camera and generate time-aligned DLT files.

## FIFA Pipeline Context
This step is typically performed after the **FIFA Bootstrap** and **FIFA Baseline** steps to ensure camera parameters are updated with any refined calibration.

---
See also: [FIFA Workflow](../../docs/fifa_workflow.md), [DLT Reconstruction](../../docs/dlt3d.md)
