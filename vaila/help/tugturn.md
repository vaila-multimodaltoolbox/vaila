# TUG and TURN

Instrumented Timed Up and Go (TUG) analysis with markerless 3D kinematics, robust
gait-event detection, deterministic spatial phase segmentation, intersegmental
Vector Coding, and Extrapolated Center of Mass (XCoM) dynamic-stability metrics.

> **Module:** `vaila/tugturn.py`
> **Authors:** Paulo R. P. Santiago, Abel G. Chinaglia
> **Version:** 0.1.2 (27 April 2026)
> **Python:** 3.12
> **License:** AGPL-3.0
> **Reference:** Chinaglia AG, Cesar GM, Santiago PRP. *Automating Timed Up and Go
> Phase Segmentation and Gait Analysis via the tugturn Markerless 3D Pipeline.*
> arXiv:2602.21425 [cs.CV], 2026. <https://arxiv.org/abs/2602.21425>
> **Project:** vailá Multimodal Toolbox — <https://github.com/vaila-multimodaltoolbox/vaila>
> **Manuscript repo:** <https://github.com/paulopreto/tugturn_GP>

## Pipeline overview

`tugturn.py` ingests a MediaPipe-style 3D CSV (33 landmarks, 1-based `p1..p33`
columns or 0-based / landmark-name schemas) plus a TOML metadata file with at
least the sampling rate `FPS`, and produces a complete clinical TUG analysis in
five stages:

1. Landmark resolution and 3D Center of Mass (CoM) estimation (Dempster-style
   weighted segmental model, hands excluded).
2. Spatial phase segmentation along the antero-posterior progression axis (Y).
3. Phase-aware Heel-Strike (HS) and Toe-Off (TO) detection (Zeni-style
   relative-distance algorithm, with kinematic and spatial masks).
4. Spatiotemporal parameters, joint angles, trunk inclination, intersegmental
   coordination (Vector Coding), and Extrapolated CoM (XCoM).
5. Reproducible output bundle: HTML reports, per-phase GIFs, JSON, and a family
   of CSV "database rows" suitable for downstream statistics or machine learning.

## Phase segmentation

The trial is partitioned into six canonical phases (skipping `stop_5s` if the
participant does not pause):

| Phase | Meaning |
| --- | --- |
| `stand`        | Sit-to-stand (chair → upright)                         |
| `first_gait`   | Forward straight-line walking toward the turn zone     |
| `stop_5s`      | Optional pause / freeze before the turn (5-s standard) |
| `turn180`      | 180° turn around the cone                              |
| `second_gait`  | Return straight-line walking toward the chair          |
| `sit`          | Stand-to-sit (upright → chair)                         |

Boundaries are computed from CoM-Z amplitude (sit/stand thresholds), CoM-Y
spatial thresholds (`y_chair`, `y_turn ± y_tol`), trunk speed plateau
(stop anchor), and shoulder-yaw rate (turn boundaries). All values are exposed
as TOML overrides (see *Configuration*).

Heel-Strike and Toe-Off events are estimated with a Zeni-style relative-distance
method, then **strictly filtered** so that any event falling outside the
First-Gait or Second-Gait windows is discarded — eliminating fictitious events
during sit-to-stand, turning, and stand-to-sit.

## Gait kinematics

For every frame the pipeline computes:

- Knee and ankle angles (Right / Left) — anatomical convention (0° extended).
- Hip angles (R / L) — using shoulder–hip–knee triangle.
- Trunk inclination relative to the global vertical (clinically relevant for
  camptocormia, festinant posture, etc.).
- Sagittal hip–knee Coupling Angle (R / L).
- 3D CoM position, velocity, and acceleration.
- Trunk, mid-trunk and per-foot mean position / velocity / acceleration.
- Step / stride length, step width, stance and swing times (with IQR outlier
  trimming on per-side aggregation), cadence, and mean velocity.

## Kin Magnitude (sagittal-plane resultant)

In the visual reports, **Kin Magnitude** is the resultant scalar magnitude
(Euclidean norm) of the motion vector in the sagittal Y/Z plane, **not** the
separate axis components:

```
|V_yz| = sqrt(Vy^2 + Vz^2)
|A_yz| = sqrt(Ay^2 + Az^2)
```

This avoids sign confusion between the AP and vertical axes and represents
absolute movement intensity.

## Intersegmental coordination — Vector Coding

### Axial Vector Coding (Trunk vs Pelvis)

During the **Turn** and **Stand** phases the trunk-pelvis coupling angle is
computed as

```
gamma = atan2(DeltaPelvis_yaw, DeltaTrunk_yaw)   (mapped to [0, 360) degrees)
```

and binned into four coordination patterns following Chang et al. (2008):

- `Proximal_Phase` — trunk dominant (0 ± 22.5° and 180 ± 22.5°)
- `In_Phase`       — both segments move in-sync (45 ± 22.5° and 225 ± 22.5°)
- `Distal_Phase`   — pelvis dominant (90 ± 22.5° and 270 ± 22.5°)
- `Anti_Phase`     — segments move in opposite directions (135 ± 22.5° and 315 ± 22.5°)

Outputs include the time-normalised coupling-angle series (101 points, 0–100 %
of phase progress), the percentage of frames per pattern, the dominant pattern,
and the Coupling Angle Variability (CAV) computed as a circular standard
deviation.

### Limb Vector Coding (Y-axis)

The same machinery is applied to limb pairs (e.g., right elbow vs right knee
along Y) during the First-Gait and Second-Gait phases, producing per-phase
arm-leg coordination summaries.

## XCoM — Extrapolated Center of Mass

Dynamic stability is summarised with the Extrapolated Center of Mass (Hof,
Gazendam & Sinke, 2005):

```
XCoM = CoM + CoM_vel / omega_0
omega_0 = sqrt(g / l)
```

with `g = 9.81 m/s^2` and `l` approximated by the mean CoM-Z height. The
*XCoM path deviation* against the straight-line trajectory is reported for the
First-Gait and Second-Gait phases and stored as
`XcoM_Deviation_First_Gait_m` / `XcoM_Deviation_Second_Gait_m`.

## Configuration (TOML)

Every CSV is paired with a TOML file with the same stem (or a single TOML
passed via `-c`). Required and optional keys:

```toml
# Subject metadata (any keys allowed; surfaced into _bd_participants.csv)
SEX = "M"
WEIGHT = 87.3
HEIGHT = 1.63
AGE = 63.0
FPS = 59.94005994005994
TURNSIDE = "R"

# Optional spatial protocol overrides (LaBioCoM walkway defaults shown)
[spatial]
y_chair     = 1.125   # m — start/end of the walkway (chair zone boundary)
y_turn      = 4.5     # m — centre of the turn zone
y_tol       = 0.5     # m — ± window around y_turn for the speed-anchor search
y_chair_tol = 0.2     # m — tolerance before y_chair to count steps inside zone
```

`y_chair_tol` can also be passed on the command line via `-y`.

## Outputs

Per-trial artefacts written to the output directory:

| File | Purpose |
| --- | --- |
| `*_tugturn_report.html`             | Static Matplotlib visual report |
| `*_tugturn_report_interactive.html` | Interactive Plotly version (XYZ overlay, gait events, stick figures) |
| `*_tugturn_data.json`               | Full structured report (metadata, phases, spatiotemporal, VC, XcoM) |
| `*_bd_results.csv`                  | One-row global summary (cadence, velocity, phase durations, VC summary) |
| `*_bd_steps.csv`                    | One row per detected step (Time_s, Y_m, Side, Phase, Step/Stride/Width, Stance/Swing) |
| `*_bd_kinematics.csv`               | One-row kinematic summary (knee max R/L, mean trunk inclination, Coupling SD R/L) |
| `*_bd_participants.csv`             | One-row TOML metadata snapshot (timestamped) |
| `*_bd_vector_coding.csv`            | Long-format coupling-angle time series for axial and limb VC |
| `*_stand.gif` … `*_sit.gif`         | One animated skeleton GIF per detected phase (segmentation QA) |

Filenames use the input CSV stem (e.g., `s26_m1_t1_*`).

## Usage

```bash
# Single trial with explicit output directory
uv run vaila/tugturn.py -i tests/tugturn/s26_m1_t1.csv \
                        -c tests/tugturn/s26_m1_t1.toml \
                        -o tests/tugturn/results/

# Single trial without -o
# Output is auto-created next to the input CSV as:
#   tugturn_result_<csv_basename>/
uv run vaila/tugturn.py -i tests/tugturn/s26_m1_t1.csv \
                        -c tests/tugturn/s26_m1_t1.toml

# Override the y_chair_tol from the CLI
uv run vaila/tugturn.py -i tests/tugturn/s26_m1_t1.csv \
                        -c tests/tugturn/s26_m1_t1.toml -y 0.15

# Batch over a directory of CSVs (each pairs with its own *.toml)
uv run vaila/tugturn.py -i tests/tugturn/ -c tests/tugturn/s26_m1_t1.toml

# GUI mode — no -i argument: a Tk file/folder dialog opens
uv run vaila/tugturn.py
```

## Reproducible test trial

A complete CLI test case ships with the repository under `tests/tugturn/`:

- `s26_m1_t1.csv` — MediaPipe 3D landmarks (≈ 27 s, 60 fps)
- `s26_m1_t1.toml` — subject metadata
- `skeleton_pose_mediapipe.json` — 33-landmark connection schema

It is exercised by 51 automated tests covering unit, integration, regression,
and end-to-end (CLI subprocess) levels:

```bash
uv run pytest tests/test_tugturn.py tests/test_tugturn_integration.py -v
```

## Citation

If you use `tugturn.py` in research, please cite the preprint:

> Chinaglia AG, Cesar GM, Santiago PRP. **Automating Timed Up and Go Phase
> Segmentation and Gait Analysis via the tugturn Markerless 3D Pipeline.**
> arXiv:2602.21425 [cs.CV], 2026.

and the parent vailá toolbox (Santiago, Cesar, Mochida, Aceros et al.,
arXiv:2410.07238).
