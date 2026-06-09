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

Optional `deadlift_parameters.txt` beside the input CSV (auto-loaded; can also
be pointed to explicitly with `--params-file/-p`).

**Preferred CSV-header format** (one row of values, mass + height + barbell +
sample rate):

```
subject_mass_kg,subject_height_m,deadlift_mass_kg,fs_hz
75.0,1.75,20.0,25.0
```

Tolerated column aliases (case-insensitive): `subject_mass`/`body_mass_kg`
/`mass_kg`/`mass`, `subject_height`/`height_m`/`height`,
`deadlift_mass`/`barbell_kg`/`weight_kg`/`weight`, and
`fs`/`fps`/`imu_fps`/`sample_rate`.

**Legacy key,value format** (still supported, one parameter per line):

```
weight,20
real_repetition_count,10
```

## Usage

**GUI (single unified form):**

```bash
uv run vaila.py
```

Then either click **Deadlift IMU** in Frame B, or click **Deadlift** and choose
*IMU (barbell accelerometer + gyroscope CSV)* in the data-source dialog. A
**single Tkinter window** opens with all inputs in one place: input directory,
output directory, params-file picker (loads numeric fields automatically),
subject mass, subject height, deadlift bar mass, sample rate, AHRS filter,
Madgwick `beta`, gyro units, and optional expected-reps count.

**CLI:**

```bash
# Auto-detects deadlift_parameters.txt beside the CSV
uv run python vaila/vaila_deadlift_imu.py -i tests/Deadlift/imu/deadlift_imu.csv

# Explicit params file (-p / --params-file)
uv run python vaila/vaila_deadlift_imu.py -i data.csv -p /path/to/deadlift_parameters.txt

# Inline overrides (win over params file)
uv run python vaila/vaila_deadlift_imu.py -i data.csv --mass 80 --height 1.80 --weight 60
uv run python vaila/vaila_deadlift_imu.py -i data.csv --filter mahony --beta 0.05
uv run python vaila/vaila_deadlift_imu.py -i data.csv --fps 100 --gyro-units rad
uv run python vaila/vaila_deadlift_imu.py -i data.csv --reps 14   # pin a known rep count

# Force the unified GUI dialog
uv run python vaila/vaila_deadlift_imu.py --gui
```

Options:

| Flag                 | Default       | Description                                          |
|----------------------|---------------|------------------------------------------------------|
| `-i / --input`       | —             | Input IMU CSV (acc_x/y/z, gyr_x/y/z required)        |
| `-o / --output`      | input parent  | Destination directory for plots/CSVs/HTML/Markdown   |
| `-p / --params-file` | auto-detect   | Path to `deadlift_parameters.txt` (any supported fmt)|
| `--filter`           | `madgwick`    | `madgwick` or `mahony`                               |
| `--beta`             | `0.1`         | Madgwick proportional gain                           |
| `--gyro-units`       | `deg`         | `deg` or `rad`                                       |
| `--fps`              | params / 25Hz | Override IMU sample rate (wins over params)          |
| `--weight`           | params / 20kg | Override barbell weight in kg                        |
| `--mass`             | params / 75kg | Override athlete body mass in kg                     |
| `--height`           | params        | Override athlete body height in meters (informational)|
| `--reps`             | auto-detect   | Force the expected repetition count                  |
| `--gui`              | —             | Force the Tkinter single-form dialog                 |

Precedence (highest first): CLI / GUI override → params file → TOML → defaults.

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
- `*_imu_quaternion_didactic_YYYYMMDD_HHMMSS.png` — **didactic quaternion
  figure** with the four components `q = [w, x, y, z]`, the norm `|q|` (should
  stay numerically at 1), the total rotation angle
  `θ = 2·arccos(|q₀|)` in degrees, and the instantaneous rotation axis
  `n = (nₓ, nᵧ, n_z)`
- `*_imu_quaternion_axis_xy_YYYYMMDD_HHMMSS.png` — XY view of the rotation
  axis `n(t)` on the unit sphere, coloured by sample index (time progression)
- `*_imu_vel_disp_YYYYMMDD_HHMMSS.png` — vertical velocity and displacement
- `*_imu_rep_bars_YYYYMMDD_HHMMSS.png` — per-rep bar charts
- `*_imu_velocity_overlay_YYYYMMDD_HHMMSS.png` — velocity profile overlay
- `*_imu_ahrs_report.html` — self-contained HTML report (with the didactic
  quaternion section and the BibTeX-ready reference list)
- `*_imu_ahrs_report.md` — Markdown sibling of the HTML report (same tables,
  same figures, same references)

### Didactic quaternion figures

A unit quaternion
`q = [q₀, q₁, q₂, q₃] = [cos(θ/2), nₓ·sin(θ/2), nᵧ·sin(θ/2), n_z·sin(θ/2)]`
represents a rotation by angle `θ` around the unit axis `n`. The two extra
quaternion figures above expose this geometry directly: the AHRS filter
produces `q(t)` sample by sample, and the report makes the abstract algebra
visible (norm check `|q| ≈ 1`, rotation angle `θ(t)`, and the trajectory of
the rotation axis `n(t)` on the unit sphere). Unlike Euler angles, quaternions
are *singularity-free* — no gimbal lock, no discontinuous unwrapping. This is
why every modern IMU/AHRS chip, every game engine and every 3D animation
system uses quaternions internally.

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
data in the book *"Modelos Matemáticos nas Ciências Não-Exatas, vol. 1"*
(Editora Blucher, 2007). Quaternions solve the *gimbal lock* singularity and are
the exact basis for the AHRS data-fusion algorithms in IMUs used in wearables and
player GPS+IMU trackers. This module is dedicated to his memory.

**Primary references (BibTeX-ready):**

- Nogueira, E. A., Martins, L. E. B., & Brenzikofer, R. (2007). *Modelos
  Matemáticos nas Ciências Não-Exatas — vol. 1.* São Paulo: Editora Blucher.
  ISBN 978-85-212-0419-0.
  <https://www.blucher.com.br/modelos-matematicos-nas-ciencias-nao-exatas-vol-1_9788521204190>
- Santiago, P. R. P. (2009). *Rotações tridimensionais em biomecânica via
  quatérnions: aplicações na análise dos movimentos esportivos.* Tese
  (Doutorado) — Universidade Estadual Paulista (Unesp), Instituto de Biociências
  de Rio Claro. <http://hdl.handle.net/11449/100404> ·
  [PDF](https://repositorio.unesp.br/server/api/core/bitstreams/41603fa7-545b-4e74-a045-57ce94885e0c/content)

**Tribute & social-media coverage:**

- Instagram post (tribute & publication history):
  <https://www.instagram.com/p/DZLogIJoFbF/>
- LinkedIn post (publication history & tribute to Prof. Brenzikofer):
  <https://www.linkedin.com/posts/paulo-roberto-pereira-santiago-132619112_hist%C3%B3rico-de-publica%C3%A7%C3%A3o-e-homenagem-ao-prof-ugcPost-7468670091845472257-3tzD/>

```bibtex
@book{nogueira2007modelos,
  title     = {Modelos matem\'aticos nas ci\^encias n\~ao-exatas - vol. 1},
  author    = {Nogueira, Eduardo Arantes and Martins, Luiz Eduardo Barreto
               and Brenzikofer, Ren\'e},
  year      = {2007},
  publisher = {Editora Blucher},
  isbn      = {9788521204190},
  url       = {https://www.blucher.com.br/modelos-matematicos-nas-ciencias-nao-exatas-vol-1_9788521204190}
}

@phdthesis{santiago2009rotaccoes,
  title    = {Rota\c{c}\~oes tridimensionais em biomec\^anica via quat\'ernions:
              aplica\c{c}\~oes na an\'alise dos movimentos esportivos},
  author   = {Santiago, Paulo Roberto Pereira},
  school   = {Universidade Estadual Paulista (Unesp),
              Instituto de Bioci\^encias de Rio Claro},
  year     = {2009},
  url      = {http://hdl.handle.net/11449/100404}
}
```
