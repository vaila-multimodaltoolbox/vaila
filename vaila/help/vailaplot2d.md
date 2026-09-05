# vailaplot2d

## Module Information

- **Category:** Visualization
- **File:** `vaila/vailaplot2d.py`
- **Version:** 0.3.121
- **Author:** Prof. Paulo Santiago
- **GUI:** yes | **CLI:** limited (helpers headless)
- **Updated:** 04 September 2026

## Description

*vailá* 2D plotting tool (Frame C → **Plot 2D**). Supports time scatter, angle-angle,
confidence intervals, boxplot, SPM, XY, and **Joint Angles** for long-format
`*_joint_angles.csv` exports from REC3D / SAM3D / Sapiens3D.

### Joint Angles

1. Click **Joint Angles**.
2. Select a `*_joint_angles.csv` (columns: `frame`, `person_id`, `joint_name`,
   `euler_x_deg`, `euler_y_deg`, `euler_z_deg`, …).
3. Choose **person_id** and **joint_name** (defaults prefer `left-knee` /
   `right-knee`). Optional **fs** (Hz): if set, X = `frame / fs` in seconds;
   otherwise X = frame index.
4. Plot shows three curves (degrees):
   - `euler_x_deg` → **Flexion/Extension**
   - `euler_y_deg` → **Abduction/Adduction**
   - `euler_z_deg` → **Internal/External Rotation**

These legend aliases are display labels for Cardan XYZ columns from the exporter.
Clinical axis/sign meaning follows `vaila/joint_kinematics.py` and the producing
pipeline — the plotter does not re-derive rotations.

Headless helpers (tests / scripts):

```python
from vaila.vailaplot2d import load_joint_angles_series, plot_joint_angles_time
import pandas as pd

df = pd.read_csv("…_joint_angles.csv")
t, series = load_joint_angles_series(df, person_id=8, joint_name="left-knee", fs=100.0)
plot_joint_angles_time(df, person_id=8, joint_name="left-knee", fs=100.0, show=True)
```

## Other plot types

**Time Scatter:** plots selected columns vs a usable `Time`/`Tempo`/`Frame`
column when present; otherwise vs sample index. Single-column force CSVs
(e.g. only `fz_N`) therefore plot correctly instead of producing 0 series.

Angle-Angle, Confidence Interval, Boxplot, SPM, XY Plot — see GUI buttons.
Clear / New Figure / Save controls manage matplotlib windows.
