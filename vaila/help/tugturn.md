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
# tugturn.py / vaila_tugtun.py

## 📋 Module Information

- **Category:** Analysis
- **File:** `vaila/tugturn.py`
- **Lines:** ~2800
- **Version:** 0.0.3
- **Author:** Paulo Roberto Pereira Santiago
- **Email:** paulosantiago@usp.br
- **GitHub:** https://github.com/vaila-multimodaltoolbox/vaila
- **GUI Interface:** ✅ Yes
- **License:** AGPLv3.0

## 📖 Description

This script provides comprehensive functionality for the **Timed Up and Go (TUG)** instrumented analysis using 3D kinematics (MediaPipe 33 landmark format). It segments the TUG test into six distinct phases and calculates detailed spatiotemporal parameters and Vector Coding metrics.

### Key Features

- **Robust Gait Event Detection**: Uses the **Zeni relative-distance algorithm** for precise Heel Strike (HS) identification, combined with pelvis velocity-based walking masks.
- **Y-Axis Spatial Segmentation**: Implements primary phase detection based on Anteroposterior (Y) displacement (LaBioCoM protocol), ensuring anatomical consistency.
- **Outlier Filtering**: Incorporates **IQR-based trimming** for spatiotemporal parameters to exclude boundary anomalies or "fictitious" steps.
- **Vector Coding Analysis**: Calculates coordination patterns during the turn phase.
- **Visual Reports**: Generates both static (Matplotlib) and interactive (Plotly) HTML reports with step-by-step GIFs for each phase.

## 🔧 Main Functions

- `detect_gait_events()` - Detects Heel Strikes (HS) and Toe Offs (TO) using Zeni and velocity heuristics.
- `segment_tug_phases()` - Segments the trial into Stand, Gait Forward, Stop/Turn, 180° Turn, Gait Back, and Sit.
- `calculate_spatiotemporal_params()` - Computes Step/Stride Length, Step Width, Stance/Swing Time, and Cadence.
- `calculate_vector_coding_turn()` - Performs Vector Coding analysis for the turn phase.
- `generate_matplotlib_report() / generate_plotly_report()` - Renders the final HTML diagnostics.

## ⚙️ Configuration Parameters

These parameters can be set in the `.toml` configuration file under the `[spatial]` section:

- **y_chair** (default: 1.125m): The Y-coordinate threshold for the chair boundary.
- **y_turn** (default: 4.5m): The target distance for the turn/pause zone.
- **y_chair_tol** (default: 0.2m): Tolerance for steps taken close to the chair during walking.
- **y_turn_tolerance** (default: 0.5m): Search window around the turn zone.

## 📁 Output Files

The script generates a results directory containing:

1. **Summary Results** (`*_bd_results.csv`): Global metrics and step sequences.
2. **Individual Steps** (`*_bd_steps.csv`): Detailed parameters for every detected step.
3. **Kinematics Data** (`*_bd_kinematics.csv`): Time-series metrics (trunk inclination, joint angles).
4. **Vector Coding** (`*_bd_vector_coding_turn.csv`): Coordination results.
5. **Phase Animations** (`*.gif`): Short clips for visual verification of each TUG phase.
6. **HTML Reports** (`*_tug_report.html`): Comprehensive dashboard with charts and GIFs.
7. **JSON Data** (`*_tug_data.json`): Complete raw results for programmatic access.

## 🚀 Usage

### CLI Usage (uv)

```bash
uv run vaila/tugturn.py -i input_data.csv -c config.toml -o output_dir -y 0.2
```

### CLI Usage (Standard Python)

```bash
python vaila/tugturn.py -i input_data.csv -c config.toml -o output_dir
```

## 💻 Requirements

- **Python 3.12.12+**
- **vailá toolbox** dependencies (NumPy, Pandas, SciPy, Matplotlib, Plotly).

## 📝 Version History

- **v0.0.3** (February 2026):
  - Implemented Zeni algorithm for gait event detection.
  - Added spatial tolerance `-y / --y_chair_tol` parameter.
  - Added IQR-based outlier filtering for robust spatiotemporal averages.
  - Refactored segmentation to use Y-axis as primary driver.
- **v0.0.1 - v0.0.2**: Initial development and testing.

---

📅 **Generated on:** February 2026  
🔗 **Part of vailá - Multimodal Toolbox**  
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
