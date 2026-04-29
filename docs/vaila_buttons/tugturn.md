# tugturn Button

The **tugturn** button launches the `tugturn.py` module for automated clinical analysis of the Timed Up and Go (TUG) test.

## Overview

`tugturn` provides a complete clinical TUG analysis using markerless 3D kinematics. It segments the test into six phases (stand, first gait, stop, turn, second gait, sit) and calculates spatiotemporal, kinematic, and stability metrics.

## Key Features

- **Automated Segmentation:** Detects TUG phases automatically based on CoM movement.
- **Advanced Metrics:**
  - Spatiotemporal parameters (cadence, velocity, step length).
  - Joint kinematics (knee, ankle, hip, trunk).
  - Intersegmental coordination via **Vector Coding**.
  - Dynamic stability via **Extrapolated Center of Mass (XCoM)**.
- **Rich Reporting:** Generates interactive HTML reports, phase-specific GIFs, and structured JSON data.

## Usage

1. Click **tugturn** in Frame B.
2. Select the MediaPipe-style 3D CSV and the corresponding TOML metadata.
3. Run the analysis to generate the clinical report bundle.

---
See also: [tugturn Help Index](../../vaila/help/tugturn.html), [arXiv:2602.21425](https://arxiv.org/abs/2602.21425)
