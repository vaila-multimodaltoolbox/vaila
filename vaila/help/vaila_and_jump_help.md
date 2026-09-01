## vailá — Vertical Jump Analysis (vaila_and_jump.py)

This guide explains how to use the Vertical Jump Analysis tool in vailá. It covers inputs, workflow, outputs, and core equations. The instructions below are in English to standardize project documentation.

### Overview
- Three modes:
  - Time-of-Flight: estimate jump height from flight time
  - Jump-Height: use measured jump height directly
  - MediaPipe Pose: read 2D pose CSVs, convert to meters, compute Center of Gravity (CG), and derive metrics
- Outputs include plots, calibrated CSVs, and an HTML report
- Team batch outputs include visual group comparison, z-score/QC tables, and optional correction of height/flight time from foot-contact frames when CoM timing is inconsistent

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
   - On macOS, the GUI launch reuses the main Tk window so prompts remain movable and focused.
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

### CLI (command line)
Same options for all data types. Use `-d` to select the mode:

- `-i` — Input: CSV file (mode 3) or directory of CSVs (modes 1 and 2)
- `-c` — Config TOML (required for mode 3 MediaPipe)
- `-o` — Output directory (optional; default: next to input, timestamped)
- `-d` — Data type: 1 = Time of Flight (directory), 2 = Jump Height (directory), 3 = MediaPipe (single CSV + config)
- `--gui` — Force GUI mode

Example (MediaPipe, mode 3):
```bash
python vaila_and_jump.py -i path/to/file.csv -c path/to/vaila_and_jump_config.toml -o path/to/output/ -d 3
```

Example (Time of Flight batch): `-i <dir> -o <out> -d 1`. Example (Jump Height batch): `-i <dir> -o <out> -d 2`.

### Calibration Check (read this first)
The pipeline recovers gravity from the CoM trajectory during flight. Airborne, the CoM is a
projectile, so a parabola fit **must** return *g* = 9.81 m/s². This is the only check that is
independent of the two values you type in, because an error in either shows up directly:

\( g_{measured} = 9.81 \cdot (fps_{entered}/fps_{true})^2 \cdot (scale_{true}/scale_{entered}) \)

If the measured *g* falls outside 85–115% of 9.81, the run is flagged `calibration_error`,
a warning is printed, and the report shows the implied frame-rate and scale corrections plus a
`*_freefall_check.png` figure overlaying the measured CoM on an ideal free-fall curve.
**The usual cause is entering the playback fps of a slow-motion clip** (e.g. 120 when the camera
captured at 240) — that quarters the measured *g* and halves every velocity.

### Jump Height: which number to use
| Field | Meaning |
| --- | --- |
| `height_com_takeoff_ref_m` | Peak CoM − CoM at foot-off. **The jump height**; QC recommends this. |
| `height_cg_method_m` / `height_com_above_standing_m` | Peak CoM above *standing* height. Larger, because the CoM already rises before the feet leave the ground. |
| `height_foot_contact_method_m` | g·t²/8 with t from last foot-off to first foot contact. Should agree with the first row. |
| `height_flight_time_method_m` | g·t²/8 over the CoM-baseline crossing interval. **Not a flight time**; reported only to expose the difference. |

### Phase Events (they are not interchangeable)
- `takeoff_frame` / `landing_frame` — CoM crosses its **standing baseline**. Not foot-off/contact.
- `takeoff_frame_foot_contact` / `landing_frame_foot_contact` — last foot-off / first foot contact.
- `takeoff_frame_kinetic` — last propulsion frame with a positive modelled GRF.

### Kinematic Analysis (Valgus/FPPA)
- **Valgus Ratio**: Knee Separation / Hip Separation. The ratio *decreases* as the knees collapse
  inward, so **< 0.8 indicates a valgus pattern**; > 1 means the knees are wider than the hips (varus).
- **FPPA**: Frontal Plane Projection Angle (2D). |FPPA| = 180° − angle(HIP-KNEE-ANKLE).
  The sign is anatomical: the knee's offset from the HIP→ANKLE line is projected onto the medial
  direction of that limb (hip → contralateral hip), giving one convention for both sides:
  **positive = valgus** (medial collapse, the ACL mechanism), **negative = varus**.
  A raw cross-product sign mirrors between limbs and cannot be used directly.
  |FPPA| > 10° indicates risk; valgus and varus are reported as distinct findings.
- **Peak excursions**: `max_valgus_angle_*` and `max_varus_angle_*` are tracked separately over the
  0.2 s post-landing window, plus `peak_fppa_deviation_*` for the largest deviation in either
  direction. A limb that never crosses into valgus is reported as such.
- **Phases Analyzed**: Squat (Propulsion Start), Initial Contact, IC+40 ms, IC+100 ms.
- **Robustness**: Uses neighbor-frame search to handle occlusion during deep squat.

### Outputs
- CSV: `<name>_jump_results_<timestamp>.csv` — scalars (mode 3)
- CSV: `<name>_jump_timeseries_<timestamp>.csv` — one row per frame (mode 3)
- CSV: `<name>_calibrated_<timestamp>.csv` — calibrated (meters) and normalized series
- PNG: 
  - Normalized diagnostic plots
  - Stick-figure phases (with Time and Jump Height annotation)
  - Valgus event analysis (with risk metrics text outside plot area)
  - Free-fall calibration check (`*_freefall_check.png`)
  - Power curve with the propulsion and flight phases shaded
- HTML: `<name>_report_<timestamp>.html` — comprehensive report with risk screening table
- Team batch: `team_jump_quality_zscores_<timestamp>.csv` and `team_jump_report_<timestamp>.html` with height distribution, Z-score matrix, and QC highlights

### Equations
- Height from time of flight: \( h = \frac{g\, t^2}{8} \)
- Takeoff velocity: \( v = \sqrt{2 g h} \)
- Potential energy: \( E_p = m g h \)
- Kinetic energy: \( E_k = \tfrac{1}{2} m v^2 \)
- Average propulsion power: \( \bar{P} = m g h / t_{prop} \)
  - \(E_k\) at take-off and \(E_p\) at the apex are the **same** energy (v is derived from h),
    so they are never summed. `total_energy_J` = \(m g h\).
  - `power_avg_propulsion_work_W` adds the work of rising from the bottom of the countermovement:
    \( (m g h + m g d_{squat}) / t_{prop} \)
- Vertical force: \( F(t) = m [a(t) + g] \) — valid **only in contact**; forced to 0 during flight,
  where the true GRF is zero
- Instantaneous power: \( P(t) = F(t)\, v(t) \), from a CoM low-pass filtered at 12 Hz
  (zero-lag Butterworth) before differentiation

### Tips
- Use the actual **capture** FPS for slow-motion videos (e.g., 240 Hz), not the playback FPS.
  **`ffprobe` will not tell you this**: a retimed slow-motion clip reports its *playback* rate
  (a real 240 fps trial in this repo reports `r_frame_rate = 30.02`). If you are unsure, run once
  and read the free-fall calibration check — it reports the implied FPS from gravity itself.
- Prefer normalized CSV for simplicity; pixel CSV is supported via scaling to meters
- Units: meters (m), seconds (s), Watts (W), Joules (J)

Version: 0.3.117  
Updated: 27 August 2026  
Author: Prof. Paulo R. P. Santiago  
License: GPL-3.0
