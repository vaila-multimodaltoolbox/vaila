# vaila-ElasticKick (VEK)

**Module:** `vaila.vek`  
**Version:** 0.3.56  
**Updated:** 23 June 2026

VEK estimates biomechanical and mechanical variables from a soccer/futsal kick
performed with elastic-band resistance.

## Inputs

- Markerless pose CSV with hip, knee, ankle, heel and foot-index landmarks.
- Optional ball tracking CSV with `frame_index`, `ball_x`, `ball_y`.
- Elastic-band calibration CSV with `length_m`, `force_n`.
- TOML configuration with athlete, trial, camera, band, ball, filter and contact settings.

## TOML Template

Generate this starter file from the CLI:

```bash
python -m vaila.vek --write-default-config vek_config.toml
```

Example:

```toml
[athlete]
id = "001"
name = "Athlete"
body_mass_kg = 75.0
shank_length_m = 0.43

[trial]
id = "01"
condition = "resisted"
dominant_limb = "right"
kicking_limb = "right"

[camera]
fps = 240.0
coordinate_mode = "normalized_shank"  # meters | pixels | normalized_shank
invert_y = true
# image_height_px = 1080
# pixels_per_meter = 520.0

[band]
anchor_x_m = 0.00
anchor_y_m = 0.25
attachment_landmark = "ankle"
slack_length_m = 1.00
calibration_csv = "vek_band_calibration.csv"
model = "pchip"  # pchip | linear | polynomial
polynomial_degree = 2
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
contact_method = "combined"
contact_threshold_m = 0.18
pre_contact_frames = 10
post_contact_frames = 10
contact_duration_frames = 3
# contact_frame = 418
```

## Band Calibration Example

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

## CLI

```bash
python -m vaila.vek \
  --input pose.csv \
  --ball ball.csv \
  --band-calibration vek_band_calibration.csv \
  --config vek_config.toml \
  --output results
```

Create a starter configuration:

```bash
python -m vaila.vek --write-default-config vek_config.toml
```

Batch/team processing:

```bash
python -m vaila.vek --batch team_dir --output results
```

## Outputs

- `*_timeseries.csv`: synchronized kinematics, band mechanics and ball variables.
- `*_summary.csv`: trial-level performance and quality-control metrics.
- `*_report.html`: dashboard with KPI cards and trend plots.
- PNG plots for force, power, band length, foot velocity, ball velocity, knee angle and contact detection.

## Scientific Note

Band tension and power are estimated from the calibration curve. They are not
direct measurements of muscle force or total kicking power. For validation,
prefer a manually confirmed impact frame.
