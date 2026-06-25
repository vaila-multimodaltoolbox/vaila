# vaila-ElasticKick (VEK)

**Module:** `vaila.vek`  
**GUI path:** `Soccer Tools -> VEK ElasticKick`  
**Version:** 0.3.56  
**Updated:** 2026-06-16

VEK is an experimental biomechanical assessment module for soccer/futsal kicks
performed under elastic-band resistance. It combines markerless pose data, ball
tracking, elastic-band calibration and event detection to estimate mechanical
and kinematic variables.

## Scientific Scope

VEK reports **elastic-band tension** and **mechanical power against the elastic
band**. These are not direct measurements of muscle force, contact force or
total kicking power. For validation or publication work, use a manually
confirmed impact frame and document the camera calibration, band calibration and
tracking pipeline.

## Required Files

### 1. Pose CSV

Expected columns include:

```text
frame_index
right_hip_x, right_hip_y
right_knee_x, right_knee_y
right_ankle_x, right_ankle_y
right_heel_x, right_heel_y
right_foot_index_x, right_foot_index_y
```

Use `left_*` columns for left-foot trials. VEK also accepts common variants such
as `frame`, `Frame`, `landmark.x` and `landmark.y`.

### 2. Ball CSV

Minimum:

```csv
frame_index,ball_x,ball_y
0,0.72,0.16
1,0.72,0.16
2,0.73,0.16
```

Optional:

```csv
frame_index,ball_x,ball_y,ball_z,confidence
0,0.72,0.16,0.00,0.91
```

If your coordinates are already metric, use `ball_x_m` and `ball_y_m`.

### 3. Elastic-Band Calibration CSV

Preferred force-length calibration:

```csv
length_m,force_n
1.00,0
1.10,55
1.20,125
1.30,210
1.40,310
1.50,425
1.60,555
```

VEK supports `pchip`, `linear` and `polynomial` models. The calibration table is
preferred over a simple Hooke-law stiffness.

## TOML Configuration Template

Generate this file automatically:

```bash
python -m vaila.vek --write-default-config vek_config.toml
```

Template:

```toml
# vaila-ElasticKick (VEK) example configuration

[athlete]
id = "001"
name = "Athlete"
body_mass_kg = 75.0
shank_length_m = 0.43

[trial]
id = "01"
condition = "resisted"
dominant_limb = "right"   # right | left
kicking_limb = "right"    # right | left

[camera]
fps = 240.0
coordinate_mode = "normalized_shank"  # meters | pixels | normalized_shank
invert_y = true
# Required for coordinate_mode = "pixels" with invert_y = true:
# image_height_px = 1080
# pixels_per_meter = 520.0

[band]
anchor_x_m = 0.00
anchor_y_m = 0.25
attachment_landmark = "ankle"  # ankle | heel | foot_index | foot_center
slack_length_m = 1.00
calibration_csv = "vek_band_calibration.csv"
model = "pchip"  # pchip | linear | polynomial
polynomial_degree = 2
# Linear fallback only when no calibration table exists:
# k_n_per_m = 850.0

[ball]
mass_kg = 0.43
initially_stationary = true

[filter]
human_cutoff_hz = 12.0
foot_cutoff_hz = 15.0
ball_cutoff_hz = 24.0
band_cutoff_hz = 12.0

[analysis]
contact_method = "combined"  # manual/configured | minimum_distance | ball_acceleration | combined
contact_threshold_m = 0.18
pre_contact_frames = 10
post_contact_frames = 10
contact_duration_frames = 3
# Recommended for validation:
# contact_frame = 418
```

## CLI Usage

Single trial:

```bash
python -m vaila.vek \
  --input pose.csv \
  --ball ball.csv \
  --band-calibration vek_band_calibration.csv \
  --config vek_config.toml \
  --output results
```

GUI:

```bash
python -m vaila.vek --gui
```

Team batch:

```bash
python -m vaila.vek --batch team_dir --output results
```

Expected team layout:

```text
team/
├── athlete_001/
│   ├── vek_config.toml
│   ├── pose.csv
│   ├── ball.csv
│   └── vek_band_calibration.csv
├── athlete_002/
└── athlete_003/
```

## Outputs

- `vek_<athlete>_<trial>_timeseries.csv`
- `vek_<athlete>_<trial>_summary.csv`
- `vek_<athlete>_<trial>_metadata.json`
- `vek_<athlete>_<trial>_report.html`
- PNG plots:
  - band force
  - band power
  - band length/extension
  - foot velocity
  - ball velocity
  - knee angle
  - contact detection trajectory

Team batch additionally writes:

- `vek_team_summary.csv`
- `vek_team_rankings.csv`
- `vek_limb_comparison.csv`
- `vek_team_report.html`

## Main Metrics

- peak band tension
- tension at impact
- maximum extension
- peak band power
- positive and negative band work
- loading/unloading rate
- peak foot velocity
- foot velocity at impact
- ball launch velocity and angle
- ball kinetic energy
- estimated ball impulse
- ball-to-foot speed ratio
- joint angles, angular velocities and ROM
- quality-control pass/fail indicators

## Current Limitations

- First implementation is 2D-first.
- Trunk inclination is reserved in the time-series schema but not yet estimated
  unless trunk landmarks are added in a future version.
- Contact detection should be reviewed visually for research-grade analysis.
- Elastic-band hysteresis is not modelled; use separate calibration tables if
  loading and unloading curves are materially different.
