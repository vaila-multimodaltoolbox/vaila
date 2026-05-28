# vaila_deadlift

- **Category:** Analysis
- **File:** `vaila/vaila_deadlift.py`
- **Version:** 0.3.46
- **Updated:** 2026-05-28
- **GUI Interface:** Yes - Frame B -> Deadlift

## Description

Deadlift/RDL biomechanical analysis for MediaPipe Pose CSV files. The tool converts normalized pose coordinates to meters using shank length, classifies the lift variant, exports a processed CSV, saves diagnostic plots, and creates an HTML report.

The wrist landmark is used as the barbell proxy because MediaPipe Pose does not track the bar directly. Use a sagittal/profile camera view for the horizontal setup checks.

## New validations

- **Arm verticality:** compares right shoulder and right wrist horizontal position at setup. Absolute offset above 5 cm flags hip position: wrist forward suggests hips too low; wrist behind suggests hips too high.
- **Bar over midfoot:** compares right wrist projection with the midpoint between right heel and right foot index. Absolute offset above 3 cm warns that the bar is not over midfoot.
- **Initial pull synchronism:** evaluates the first 15% of the concentric pull. If knee extension velocity is more than 2x hip opening velocity, the report flags a critical early-hip-rise / good-morning pattern.

## Existing metrics

- Stance width ratio
- Knee flexion
- Tibia angle relative to ground
- Spine angular deviation from setup
- Bar path proximity to tibia/knee line
- Cervical deviation
- Deadlift variant classification: stiff-legged, RDL, conventional, or mixed

## Inputs

MediaPipe Pose CSV files with landmark columns such as `right_shoulder_x`, `right_wrist_x`, `right_heel_x`, `right_foot_index_x`, `right_knee_x`, and matching `_y` columns. The script creates metric columns with `_m` suffix after calibration.

Optional config file: `vaila_deadlift_config.toml` beside the input CSV or in `vaila/`:

```toml
[deadlift_context]
fps = 30
shank_length_m = 0.40
mass_kg = 75
```

## Usage

GUI:

```bash
uv run vaila.py
```

Then click **Deadlift** in Frame B and select a folder containing MediaPipe CSV files.

CLI:

```bash
uv run python vaila/vaila_deadlift.py -i path/to/pose.csv -o path/to/output
```

## Outputs

- `*_deadlift_kinematics_YYYYMMDD_HHMMSS.csv` with frame-by-frame metrics
- `*_lower_limb_YYYYMMDD_HHMMSS.png`
- `*_safety_metrics_YYYYMMDD_HHMMSS.png`
- `*_biomechanical_report.html`
