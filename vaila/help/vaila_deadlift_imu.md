# vaila_deadlift_imu

- **Category:** Analysis
- **File:** `vaila/vaila_deadlift_imu.py`
- **Version:** 0.3.48
- **Updated:** 2026-06-09
- **GUI Interface:** Yes — Frame B → **Deadlift IMU** (B6_r7_c1), or **Deadlift** (B5_r6_c5) → choose *IMU (AHRS)* in the data-source dialog

## Description

Dedicated IMU-only Deadlift / RDL analysis using AHRS sensor fusion. This is
the companion to [`vaila_deadlift`](vaila_deadlift.md), specialised for
barbell-mounted inertial sensors. Unlike the simple "project the raw
accelerometer onto the static gravity axis" approach, this module tracks the
sensor orientation as a quaternion and rotates the accelerometer into the
**Earth frame** before integrating velocity / displacement. The result is
robust to arbitrary barbell rotations during the pull.

Two industry-standard AHRS filters are ported directly from the
[x-io Technologies](https://x-io.co.uk/open-source-imu-and-ahrs-algorithms/)
open-source C reference:

- **Madgwick** AHRS — gradient-descent quaternion update (Madgwick S.O.H.,
  2010). IMU branch (no magnetometer) and full AHRS branch (with
  magnetometer) are both implemented.
- **Mahony** AHRS — complementary-style proportional + integral quaternion
  update (Mahony R., Hamel T., Pflimlin J.-M., 2008). IMU branch.

Reference code lives under
`tests/Deadlift/madgwick_algorithm_c/MadgwickAHRS/` and `MahonyAHRS/`
(C sources from the [xioTechnologies/Fusion](https://github.com/xioTechnologies/Fusion)
repository). A friendly walk-through is also available on the
[Medium Madgwick explanation](https://medium.com/@k66115704/imu-madgwick-filter-explanation-556fbe7f02e3).

## Pipeline

1. Detect a low-motion **static window** at the start of the recording and
   estimate gyroscope bias + the gravity vector in the sensor frame.
2. Initialise the quaternion from gravity using
   `roll = atan2(ay, az)`, `pitch = atan2(-ax, sqrt(ay² + az²))`, `yaw = 0`.
3. Run the AHRS filter sample-by-sample to track orientation as a quaternion
   `q = [w, x, y, z]` (sensor frame relative to Earth frame).
4. Rotate the raw accelerometer to the Earth frame, subtract the measured
   static gravity → **linear acceleration** in Earth coordinates.
5. Detect repetitions from the filtered vertical linear acceleration
   (heavy ~1.2 Hz low-pass for detection, light ~5 Hz low-pass for metrics).
6. Per-rep integration with ZUPT-style linear drift correction
   (velocity and displacement re-zeroed at start/end of each rep).
7. Per-rep metrics — peak / mean velocity, peak / mean power, work, ROM,
   peak / mean force, impulse — using either the barbell weight or the total
   system mass.
8. PNG plots and a self-contained HTML report.

## Inputs

CSV with at minimum:

```
acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z
```

Optional magnetometer columns `mag_x, mag_y, mag_z` enable Madgwick's full
AHRS branch (with heading correction).

| Column            | Units                                         |
|-------------------|-----------------------------------------------|
| `acc_x/y/z`       | m/s² (gravity included)                       |
| `gyr_x/y/z`       | deg/s by default; use `--gyro-units rad` for rad/s |
| `mag_x/y/z` (opt) | any consistent unit (normalised internally)   |

Optional config file `vaila_deadlift_config.toml` beside the input CSV or in
`vaila/`:

```toml
[deadlift_context]
imu_fps = 25.0          # sample rate, Hz
weight_kg = 20.0        # barbell weight
mass_kg = 75.0          # athlete body mass (for total-mass power)
use_total_mass_for_power = true
```

Optional `deadlift_parameters.txt` beside the input CSV (auto-loaded):

```
weight,20
real_repetition_count,10
```

## Usage

**GUI:**

```bash
uv run vaila.py
```

Then either click **Deadlift IMU** in Frame B, or click **Deadlift** and choose
*IMU (barbell accelerometer + gyroscope CSV)* in the data-source dialog. Select
a folder of IMU CSVs.

**CLI:**

```bash
uv run python vaila/vaila_deadlift_imu.py -i tests/Deadlift/deadlift_imu.csv
uv run python vaila/vaila_deadlift_imu.py -i data.csv --filter mahony --beta 0.05
uv run python vaila/vaila_deadlift_imu.py -i data.csv --fps 100 --gyro-units rad
uv run python vaila/vaila_deadlift_imu.py -i data.csv --weight 60 --mass 75
uv run python vaila/vaila_deadlift_imu.py -i data.csv --reps 14   # pin a known rep count
```

Options:

| Flag              | Default      | Description                                         |
|-------------------|--------------|-----------------------------------------------------|
| `--filter`        | `madgwick`   | `madgwick` or `mahony`                              |
| `--beta`          | `0.1`        | Madgwick proportional gain                          |
| `--gyro-units`    | `deg`        | `deg` or `rad`                                      |
| `--fps`           | TOML / 25 Hz | Override IMU sample rate                            |
| `--weight`        | TOML / params| Barbell weight (kg)                                 |
| `--mass`          | TOML         | Athlete body mass (kg)                              |
| `--reps`          | auto-detect  | Force the expected repetition count                 |
| `--gui`           | —            | Force Tkinter file picker                           |

### Repetition detection

Repetitions are segmented automatically from concentric vertical-acceleration
peaks (one rep per peak, boundaries at the eccentric bottom between peaks), with
leading/trailing **static-window rejection** so the barbell-at-rest periods at
the start/end of a set are not counted. The shared `deadlift_parameters.txt`
rep count is **not** used to trim the AHRS detection (it often describes a
different capture); use `--reps N` to pin a known count. On the bundled
`tests/Deadlift/deadlift_imu.csv` example (14 reps), the detector reports 14.

## Outputs

Each input CSV produces a sibling folder named
`vaila_deadlift_imu_ahrs_YYYYMMDD_HHMMSS/<input_basename>/` containing:

- `*_imu_ahrs_processed_YYYYMMDD_HHMMSS.csv` — per-sample time series
  (quaternion, Euler angles, Earth-frame acceleration, linear acceleration,
  vertical velocity, displacement)
- `*_imu_ahrs_rep_metrics_YYYYMMDD_HHMMSS.csv` — per-rep summary
  (peak / mean velocity, power, work, ROM, peak / mean force, impulse)
- `*_imu_raw_acc_axes_YYYYMMDD_HHMMSS.png` — raw `acc_x` / `acc_y` / `acc_z`
  (gravity included) with the dominant (likely vertical) axis highlighted and
  the detected repetition peaks marked
- `*_imu_ahrs_orientation_YYYYMMDD_HHMMSS.png` — roll / pitch / yaw, Earth
  acceleration components, vertical linear acceleration
- `*_imu_vel_disp_YYYYMMDD_HHMMSS.png` — vertical velocity and displacement
- `*_imu_rep_bars_YYYYMMDD_HHMMSS.png` — per-rep bar charts
- `*_imu_velocity_overlay_YYYYMMDD_HHMMSS.png` — velocity profile overlay
- `*_imu_ahrs_report.html` — self-contained HTML report

## Reference

- Madgwick, S. O. H. (2010). *An efficient orientation filter for inertial
  and inertial/magnetic sensor arrays.*
- Mahony, R., Hamel, T., & Pflimlin, J.-M. (2008). *Nonlinear complementary
  filters on the special orthogonal group.* IEEE Transactions on Automatic
  Control, 53(5), 1203–1218.
- x-io Technologies open-source AHRS reference C code:
  <https://x-io.co.uk/open-source-imu-and-ahrs-algorithms/>
- xioTechnologies/Fusion repository (modern C/C++ port):
  <https://github.com/xioTechnologies/Fusion/tree/main>
- x-IMU3 user manual + downloads:
  <https://x-io.co.uk/x-imu3/#downloads>
- Friendly tutorial on the Madgwick filter:
  <https://medium.com/@k66115704/imu-madgwick-filter-explanation-556fbe7f02e3>

## Publication History & Tribute to Prof. René Jean Brenzikofer

The quaternion-based 3D kinematic modeling used by this module traces back to a
close academic collaboration with Professor **René Jean Brenzikofer** (UNICAMP),
which led to the first publication of quaternion-based three-dimensional modeling
data in the book *"Modelos Matemáticos nas Ciências Não-Exatas"* (Editora
Blucher, 2007). Quaternions solve the *gimbal lock* singularity and are the exact
basis for the AHRS data-fusion algorithms in IMUs used in wearables and player
GPS+IMU trackers. This module is dedicated to his memory.

- Tribute & publication history: <https://lnkd.in/eZUcgbNT>
- Doctoral thesis (Quaternions in Biomechanics): <https://lnkd.in/eSjuDjnw>
- Reference book (Editora Blucher, 2007), *Modelos Matemáticos nas Ciências
  Não-Exatas*: <https://lnkd.in/ejKpHWzK>
