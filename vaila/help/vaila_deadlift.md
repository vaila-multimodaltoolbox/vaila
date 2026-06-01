# vaila_deadlift

- **Category:** Analysis
- **File:** `vaila/vaila_deadlift.py`
- **Version:** 0.3.46
- **Updated:** 2026-05-29
- **GUI Interface:** Yes - Frame B -> Deadlift

## Description

Deadlift/RDL biomechanical analysis supporting two data types:

1. **IMU data** (accelerometer + gyroscope CSV from barbell-mounted sensors)
2. **MediaPipe Pose** CSV files with landmark coordinates

The script auto-detects the data type from CSV column headers and routes to the appropriate pipeline.

### IMU Pipeline

- Automatic repetition detection from filtered vertical acceleration
- Per-rep velocity estimation via numerical integration with drift correction
- Per-rep power (P = F × v) and work (W = ∫P dt) from barbell weight
- Cadence metrics (reps/min, inter-rep intervals, timing variability)
- Center of mass trajectory (barbell displacement)
- Rep-to-rep comparison: max, min, range, CV%, best/worst rep identification

### MediaPipe Pipeline

- Converts normalized coordinates to meters using shank length calibration
- Multi-repetition detection from shoulder/hip vertical displacement
- Whole-body center of mass estimation (De Leva 1996 segment proportions)
- Foot spread (ankle-to-ankle) and knee spread (knee-to-knee) distances
- Hand-to-foot horizontal distance
- Per-rep power & work from COM vertical velocity and body mass
- Cadence and rep-to-rep comparison statistics
- Deadlift variant classification: stiff-legged, RDL, conventional, or mixed
- Form quality validations (arm verticality, bar over midfoot, pull synchronism)

## Form Validations (MediaPipe)

- **Arm verticality:** compares right shoulder and right wrist horizontal position at setup. Absolute offset above 5 cm flags hip position.
- **Bar over midfoot:** compares right wrist projection with the midpoint between right heel and right foot index. Absolute offset above 3 cm warns.
- **Initial pull synchronism:** evaluates the first 15% of the concentric pull. If knee extension velocity > 2x hip opening velocity, flags critical early-hip-rise / good-morning pattern.

## Metrics

- Stance width ratio
- Knee flexion (left/right)
- Tibia angle relative to ground
- Spine angular deviation from setup
- Bar path proximity to tibia/knee line
- Cervical deviation
- Foot spread and knee spread (meters)
- Hand-to-foot horizontal offset (meters)
- Center of mass trajectory (x, y)
- Per-rep: peak/mean velocity, peak/mean power, work, ROM, force
- Cadence: reps/min, mean/std rep duration, interval variability
- Rep comparison: max, min, range, CV%, best/worst rep

## Inputs

**IMU data:** CSV with columns `acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z`.

**MediaPipe data:** CSV with landmark columns such as `right_shoulder_x`, `right_wrist_x`, `right_heel_x`, `right_foot_index_x`, `right_knee_x`, and matching `_y` columns.

Optional config file: `vaila_deadlift_config.toml` beside the input CSV or in `vaila/`:

```toml
[deadlift_context]
fps = 30
shank_length_m = 0.40
mass_kg = 75
weight_kg = 20
imu_fps = 25
```

Optional parameters file: `deadlift_parameters.txt` beside the input CSV (auto-reads barbell weight).

## Usage

GUI:

```bash
uv run vaila.py
```

Then click **Deadlift** in Frame B and select a folder containing CSV files (IMU or MediaPipe).

CLI:

```bash
uv run python vaila/vaila_deadlift.py -i path/to/data.csv -o path/to/output
```

## Outputs

### IMU mode

- `*_imu_processed_YYYYMMDD_HHMMSS.csv` with time series (vert. acc, velocity, displacement)
- `*_rep_metrics_YYYYMMDD_HHMMSS.csv` with per-rep summary
- `*_imu_timeseries_YYYYMMDD_HHMMSS.png` (acceleration, velocity, displacement)
- `*_imu_rep_bars_YYYYMMDD_HHMMSS.png` (per-rep bar charts)
- `*_imu_velocity_overlay_YYYYMMDD_HHMMSS.png` (velocity profiles stacked)
- `*_imu_biomechanical_report.html`

### MediaPipe mode

- `*_deadlift_kinematics_YYYYMMDD_HHMMSS.csv` with frame-by-frame metrics
- `*_rep_metrics_YYYYMMDD_HHMMSS.csv` with per-rep summary
- `*_lower_limb_YYYYMMDD_HHMMSS.png`
- `*_safety_metrics_YYYYMMDD_HHMMSS.png`
- `*_body_distances_YYYYMMDD_HHMMSS.png` (foot/knee spread, hand-foot, COM)
- `*_rep_bars_YYYYMMDD_HHMMSS.png` (per-rep bar charts)
- `*_biomechanical_report.html`
