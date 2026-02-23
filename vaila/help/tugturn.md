# TUG and TURN

Instrumented Timed Up and Go (TUG) analysis with 3D kinematics, robust gait-event detection, phase segmentation, Vector Coding, and XCoM-based stability metrics.

## Pipeline and segmentation

`tugturn.py` processes MediaPipe 3D CSV files and segments each trial into:

- `stand`
- `gait_forward`
- `turn180`
- `gait_back`
- `sit`

Segmentation combines temporal anchors and TOML spatial thresholds (`y_chair`, `y_turn`, `y_chair_tol`) to improve robustness in clinical data.

Gait events (HS/TO) are estimated with a relative-distance method (Zeni-style approach), then filtered by expected phase to reduce fictitious events.

## Gait kinematics

The script computes:

- Knee and ankle angles (R/L)
- Trunk inclination
- Trunk and foot kinematics in Y/Z (position, velocity, acceleration)
- Spatiotemporal parameters (global and phase-specific)

## Kin Magnitude (2D Kinematics Magnitude)

In the charts, **Kin Magnitude** means the resultant scalar magnitude (Euclidean norm) in the sagittal YZ plane, not separate Y and Z components.

- Velocity magnitude: `|V_yz| = sqrt(Vy^2 + Vz^2)`
- Acceleration magnitude: `|A_yz| = sqrt(Ay^2 + Az^2)`

This avoids sign confusion from separate axes and represents the absolute movement intensity.

## Coupling Angle and Vector Coding

### Axial Vector Coding (trunk vs pelvis)

During turn and stand phases, axial coordination is computed as:

`gamma = atan2(DeltaPelvis, DeltaTrunk)` normalized to `[0, 360)`.

Patterns:

- `Proximal_Phase`
- `In_Phase`
- `Distal_Phase`
- `Anti_Phase`

Outputs include the time-normalized coupling-angle series, percentage by pattern, and CAV (Coupling Angle Variability).

### Limb/sagittal coupling

The script also computes sagittal coupling metrics, including hip-knee coupling angles (`Coupling_Angle_Hip_Knee_R/L`) and limb vector coding in AP (Y) during gait-forward and gait-back.

## XCoM (Extrapolated Center of Mass)

Computed as:

`XCoM = CoM + CoM_vel / omega_0`, where `omega_0 = sqrt(g / l)`.

`l` is approximated by mean `CoM_z`. XCoM path deviation is used as a dynamic stability metric for gait forward/back phases.

## Configuration (TOML)

Use `[spatial]` parameters:

- `y_chair`: chair position along Y
- `y_turn`: turn target position along Y
- `y_chair_tol`: tolerance near chair

## Outputs

Typical per-trial outputs:

- `*_tug_report.html`
- `*_tug_report_interactive.html`
- `*_bd_results.csv`
- `*_bd_steps.csv`
- `*_bd_kinematics.csv`
- Vector-coding metrics/time-series CSVs
- Phase GIFs for segmentation QA

## Usage

```bash
# CLI with explicit output
uv run vaila/tugturn.py -i video_data.csv -c config.toml -o results/

# CLI without -o
# Output is auto-created next to each CSV:
# result_tugturn_<csv_basename>
uv run vaila/tugturn.py -i video_data.csv -c config.toml

# Batch directory input
uv run vaila/tugturn.py -i tests/tugturn/ -c tests/tugturn/s26_m1_t1.toml
```
