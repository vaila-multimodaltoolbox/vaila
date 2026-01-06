## vailá — Vertical Jump Analysis (vaila_and_jump.py)

This guide explains how to use the Vertical Jump Analysis tool in vailá. It covers inputs, workflow, outputs, and core equations. The instructions below are in English to standardize project documentation.

### Overview
- Three modes:
  - Time-of-Flight: estimate jump height from flight time
  - Jump-Height: use measured jump height directly
  - MediaPipe Pose: read 2D pose CSVs, convert to meters, compute Center of Gravity (CG), and derive metrics
- Outputs include plots, calibrated CSVs, and an HTML report

### Accepted MediaPipe Inputs (CSV)
- File naming (vailá style): landmark columns like `nose_x`, `nose_y`, `right_ankle_x`, `right_ankle_y`, etc.
- You may use either of the CSVs exported by Markerless 2D Analysis:
  1) Normalized CSV: `*_mp_norm.csv` (x,y in [0..1]) — recommended
  2) Pixel CSV: `*_mp_pixel.csv` (x,y in pixels)
- The script converts coordinates to meters using a shank-length scale factor, so both forms are supported. For the most predictable behavior, prefer the normalized CSV.

### Coordinate System
- Biomechanical convention: y increases upward
- For MediaPipe inputs, the script reorients y to match this convention and then converts all coordinates to meters
- CG is normalized relative to the average CG over the initial frames (default: frames 10–20)

### Quick Start
1) Run the tool
   - From vailá GUI or: `python vaila/vaila_and_jump.py`
2) Choose the directory with your CSV files
3) Select the data type: (1) Time of Flight, (2) Jump Height, or (3) MediaPipe
4) MediaPipe mode requires subject constants (asked once per batch):
   - Mass (kg), Video FPS, Shank length (m)
     - **Note**: FPS can be a decimal (float) for high-speed cameras.
   - To avoid prompts, create `vaila_and_jump_config.toml` in `vaila/` or `vaila/models/`:
     ```toml
     [jump_context]
     mass_kg = 75.0
     fps = 240.0
     shank_length_m = 0.40
     ```
5) Review outputs in the generated timestamped folder

### Kinematic Analysis (Valgus/FPPA)
- **Valgus Ratio**: Knee Separation / Hip Separation. (< 0.8 indicates risk)
- **FPPA**: Frontal Plane Projection Angle (2D). (> 10° indicates risk)
- **Phases Analyzed**: Squat (Propulsion Start) and Landing Sequences.
- **Robustness**: Uses neighbor-frame search to handle occlusion during deep squat.

### Outputs
- CSV: `<name>_jump_metrics_<timestamp>.csv` — summary metrics
- CSV: `<name>_calibrated_<timestamp>.csv` — calibrated (meters) and normalized series
- CSV: `<name>_jump_database_<timestamp>.csv` — one-row database with all metrics
- PNG: 
  - Normalized diagnostic plots
  - Stick-figure phases (with Time and Jump Height annotation)
  - Valgus event analysis (with risk metrics text outside plot area)
- HTML: `<name>_report_<timestamp>.html` — comprehensive report with risk screening table

### Equations
- Height from time of flight: \( h = \frac{g\, t^2}{8} \)
- Takeoff velocity: \( v = \sqrt{2 g h} \)
- Potential energy: \( E_p = m g h \)
- Kinetic energy: \( E_k = \tfrac{1}{2} m v^2 \)
- Average propulsion power: \( \bar{P} = (E_k + E_p) / t_{prop} \)
- Vertical force: \( F(t) = m [a(t) + g] \)
- Instantaneous power: \( P(t) = F(t)\, v(t) \)

### Tips
- Use the actual capture FPS for slow-motion videos (e.g., 240 Hz)
- Prefer normalized CSV for simplicity; pixel CSV is supported via scaling to meters
- Units: meters (m), seconds (s), Watts (W), Joules (J)

Author: Prof. Paulo R. P. Santiago  
License: GPL-3.0
