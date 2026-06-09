# vaila_deadlift

- **Category:** Analysis
- **File:** `vaila/vaila_deadlift.py`
- **Version:** 0.3.48
- **Updated:** 2026-06-09
- **GUI Interface:** Yes - Frame B -> **Deadlift** (B5_r6_c5). The button opens a data-source dialog: *Kinematics (MediaPipe)* routes here; *IMU (AHRS)* routes to [`vaila_deadlift_imu`](vaila_deadlift_imu.md).

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

- Converts normalized coordinates to meters using shank length calibration, or preserves existing calibrated `_m` columns
- Builds bar-height proxy columns (`bar_marker_y_m`, `bar_height_m`) from the averaged left/right wrist Y markers
- Multi-repetition detection from averaged wrist/bar-height phases: low-start counts the first standing maximum; standing-start skips setup and counts the next standing maximum after the low phase
- Whole-body center of mass estimation (De Leva 1996 segment proportions)
- Foot spread (ankle-to-ankle) and knee spread (knee-to-knee) distances
- Hand-to-foot horizontal distance
- Per-rep power & work from COM vertical velocity and configured load mass
- Cadence and rep-to-rep comparison statistics
- Deadlift variant classification: stiff-legged, RDL, conventional, or mixed
- Form quality validations (arm verticality, bar over midfoot, pull synchronism)

## Form Validations (MediaPipe)

- **Arm verticality:** compares shoulder midpoint and averaged wrist/bar marker horizontal position at setup. Absolute offset above 5 cm flags hip position.
- **Bar over midfoot:** compares averaged wrist/bar marker projection with the midpoint between right heel and right foot index. Absolute offset above 3 cm warns.
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
- Bar marker trajectory and bar height from averaged wrist Y
- Per-rep: count frame/time, bottom frame/time, peak/mean velocity, peak/mean power, work, COM ROM, bar ROM, force/load mass
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
use_total_mass_for_power = true
imu_fps = 25
```

Optional legacy file: `deadlift_parameters.txt` beside the input CSV or one of its nearby parent directories. It auto-reads `real_repetition_count` for mismatch reporting only; subject/load values should use `subject_parameters.txt` or CLI/GUI overrides.

Kinematics/camera parameter file (`kinematics_parameter.txt`) can be selected in GUI or passed by CLI. The parser reads values such as `Recommended Hz`, `Display FPS`, `Avg FPS`, `Frames`, `Duration`, and `Resolution`. FPS/Hz is used to create `time_s`, and MediaPipe plots use seconds on the x-axis.

Subject/load parameter file (`subject_parameters.txt`) can be selected in GUI or passed by CLI. CSV format example:

```csv
subjectmass_kg,shank_len_meter,loadmass_kg
75,0.44,20
```

## Usage

GUI:

```bash
uv run vaila.py
```

Then click **Deadlift** in Frame B. A data-source dialog appears:

- **Kinematics (MediaPipe pose CSV)** — runs this module (`vaila_deadlift.py`).
- **IMU (barbell accelerometer + gyroscope CSV)** — runs the AHRS companion [`vaila_deadlift_imu`](vaila_deadlift_imu.md).

Select the matching option, then choose a folder containing the CSV files.

CLI:

```bash
uv run python vaila/vaila_deadlift.py -i path/to/data.csv -o path/to/output
uv run python vaila/vaila_deadlift.py -i path/to/data.csv -o path/to/output \
  --kinematics-params path/to/kinematics_parameter.txt \
  --subject-params path/to/subject_parameters.txt \
  --fps 59.94 --mass-kg 75 --shank-length-m 0.44 --barbell-mass-kg 20 \
  --gif-duration-s 1.5
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
- `*_calibrated_YYYYMMDD_HHMMSS.csv` with calibrated marker, COM, and bar-height columns
- `*_rep_metrics_YYYYMMDD_HHMMSS.csv` with per-rep summary
- `*_lower_limb_YYYYMMDD_HHMMSS.png`
- `*_safety_metrics_YYYYMMDD_HHMMSS.png`
- `*_body_distances_YYYYMMDD_HHMMSS.png` (time-based foot/knee spread, hand-foot, COM, bar height, peak markers, and low-phase valley intervals)
- `*_rep_bars_YYYYMMDD_HHMMSS.png` (per-rep bar charts)
- `*_stickfigures_phases_YYYYMMDD_HHMMSS.png` with sequential low/standing keyframes, counter, frame, and time
- `*_deadlift_anim_YYYYMMDD_HHMMSS.gif` with slower sequential low/standing keyframes, counter, frame, and time
- `*_biomechanical_report.html`
