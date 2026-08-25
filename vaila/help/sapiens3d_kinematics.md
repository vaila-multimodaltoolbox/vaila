# Sapiens2 3D Kinematics (`sapiens3d_kinematics.py`)

## Module information

- **Category:** Data Files
- **Version:** 0.3.113
- **Updated:** 2026-08-24
- **GUI:** Frame C → Data Files → **Sapiens2 3D Kinematics** (`C_A_r4_c2`)
- **CLI:** Yes

## Purpose

Computes functional 3D lower-limb kinematics — pelvis / thigh(L,R) /
shank(L,R) / foot(L,R) local coordinate systems, relative **Hip**, **Knee**,
and **Ankle** joint rotation matrices, quaternions, and Euler/Cardan angles —
from a Sapiens2 REC3D `.c3d` file.

Pipeline: Sapiens2 2D keypoints → REC3D (DLT triangulation) → C3D with
generic `p1..pN` point labels. Neither `rec3d.py` nor `rec3d_one_dlt3d.py`
carries keypoint names into the C3D writer, so semantic identity is lost at
export time unless it can be recovered afterward — which is exactly what
this module's landmark resolver does before computing anything.

## Keypoint mapping — read this before trusting any output

A REC3D-exported C3D never stores which anatomical landmark each `pN` is.
The module resolves the mapping in this order, and refuses to guess if none
apply:

1. **Direct label match** — if the C3D's own point labels are already
   semantic (`left_hip`, `l_hip`, `LHip`, ...), match them directly.
2. **Canonical 308-keypoint order** — if the true point count (`POINT:LABELS`
   merged with its `LABELS2`/`LABELS3`/... continuation groups via
   `vaila.readc3d_export.get_point_labels` — a naive read of `POINT:LABELS`
   alone silently truncates at the C3D format's 255-entry-per-group cap) is
   exactly **308**, `pN` is treated as the 1-based index into the canonical
   Sapiens2 "Sociopticon" order shipped at
   `vaila/skeletons/sapiens2_goliath308.json`. This is the common case for a
   full-topology Sapiens2 REC3D export.
3. **Explicit `--keypoint-map`** — if the point count matches no known
   canonical topology in the repository, the module raises
   `LandmarkResolutionError` naming the missing anatomical slots instead of
   fabricating a mapping. Supply a JSON file such as:

   ```json
   {"left_hip": "p10", "right_hip": "p11", "left_knee": "p12", ...}
   ```

   (point labels or 0-based integer indices are both accepted).

## Scientific limitation — functional markerless model, not Plug-in Gait

Sapiens2 provides no equivalent of Vicon's THI/TIB wands, femoral/tibial
epicondyle markers, or an anatomical calibration trial. Without those, the
longitudinal (axial-rotation) axis of the thigh and shank is not directly
observable from hip/knee/ankle joint centers alone. Segment frames are built
from joint centers only (Gram-Schmidt) — a **functional/surrogate** model:

- **flexion/extension** (first Euler rotation, both sequences) is the most
  directly observable component;
- **ab/adduction** (frontal plane) is a functional estimate;
- **internal/external rotation** (transverse plane / axial) is a surrogate
  estimate and should **not** be interpreted as anatomically identical to
  Plug-in Gait's axial rotation.

Thigh and shank frames no longer depend on the adjacent segment (thigh does
not use ankle; shank does not use hip) — each derives its mediolateral axis
from the pelvis (thigh) or from the already-built proximal segment's own Y
axis, chain-propagated distally (shank). This removes the straight-leg
(collinear hip-knee-ankle) singularity the joint-center-only Gram-Schmidt
construction previously had. A residual, anatomically distinct singularity
remains if the thigh's own long axis (hip→knee) happens to be parallel to
the pelvis's mediolateral axis (femur pointing directly sideways) — a
degenerate hip-abduction pose, not triggered by knee flexion.

The pelvis frame requires a superior landmark (neck, or the midpoint of both
shoulders). When neither is available, the module does **not** fabricate a
plausible-looking pelvis orientation: it emits `NaN` for the pelvis's
anterior/vertical axes (and therefore for `R_hip`) and records a QC warning
in `kinematics_metadata.json`'s `qc_warnings` list and on stderr. The
pelvis's mediolateral axis is unaffected and still usable as the thigh
frames' secondary reference.

Reference: Wu G et al., *J Biomech* 2002;35:543-548, DOI:
10.1016/S0021-9290(01)00222-6 (ISB joint-coordinate-system convention,
followed in spirit — not letter, since no ASIS/PSIS/epicondyle markers are
available); Grood & Suntay (1983) for the general joint-coordinate-system
concept.

## Quaternion convention

vailá already has three mutually inconsistent quaternion conventions across
`rotation.py` (docstring claims scalar-first `(w,x,y,z)` but the code
actually returns scipy's scalar-last `[x,y,z,w]` — a known, unfixed repo
bug), `joint_kinematics.py` (scalar-first `[w,x,y,z]`, correctly documented),
and the IMU modules (scalar-first `[w,x,y,z]`). **This module** uses scipy's
native scalar-last `[x, y, z, w]` (`Rotation.as_quat()` directly, with no
docstring/code mismatch), with per-DOF temporal sign continuity enforced
(`dot(q[t], q[t-1]) < 0 → q[t] *= -1`). Output CSV columns are named
`..._qx, ..._qy, ..._qz, ..._qw` so the convention is unambiguous without
reading this doc.

## Euler/Cardan conventions

Two sequences are written for every segment/joint (scipy intrinsic,
uppercase-letter sequence strings):

- `functional_xyz` — `XYZ` for all three joints (this module's own default).
- `vicon_compatible` — Hip `YXZ`, Knee `YXZ`, Ankle `YZX` (matches current
  Plug-in Gait lower-body rotation orders; the axis **assignment** to
  anatomical flexion/ab-adduction/rotation still carries the functional/
  surrogate caveat above — matching the rotation *order* is not the same as
  matching the anatomical calibration).

## Coordinate frame definitions

| Segment | Origin | Primary axis | Secondary reference |
|---|---|---|---|
| Pelvis | mid-hip | Y = left_hip − right_hip | neck (or mid-shoulder) − mid_hip for the up direction |
| Thigh | hip | Z = knee − hip | pelvis mediolateral axis (left_hip − right_hip) — independent of knee/ankle |
| Shank | knee | Z = ankle − knee | the thigh frame's own Y axis, chain-propagated — independent of hip/heel/toe |
| Foot | ankle | X = toe − heel | (knee − ankle) component orthogonal to X |

**Left/right sign convention:** each segment is built once from an
unsigned, side-independent secondary reference, then the right-side frame
is corrected by a single fixed local 180° rotation (`R_right = R_raw @
diag(1, −1, −1)` for thigh/shank, `diag(−1, 1, −1)` for foot — always a
proper rotation, since both are valid signed permutations of the identity
with determinant +1). This was verified against a synthetic sagittal-mirror
test: a naive per-side sign flip folded into the auxiliary vector before
the cross product (the previous convention) reports matching flexion/
extension between mirrored left/right legs but inconsistent signs on
ab/adduction, varus/valgus, and axial-rotation/inversion-eversion; the
fixed post-hoc rotation reports all three Euler components consistently
for both sides (`tests/test_sapiens3d_kinematics.py::
test_mirror_left_right_report_consistent_anatomical_signs`).

Relative joint rotations:

```text
R_hip   = R_pelvis.T @ R_thigh
R_knee  = R_thigh.T @ R_shank
R_ankle = R_shank.T @ R_foot
```

Every rotation matrix is projected onto SO(3) via SVD/polar decomposition
(`project_to_so3`) before use, and every frame's determinant and
orthogonality error (`||RᵀR − I||`) are written to the angle CSVs as QC
columns.

## Raw and neutral-zeroed orientations

Pass `--neutral-frames start:end` (0-based, half-open) to additionally
compute a neutral pose (quaternion-averaged over that frame range, Markley
et al. 2007) and zero every segment/joint rotation against it
(`R_zeroed[t] = R_neutral.T @ R[t]`). Raw and neutral-zeroed outputs are
written as separate CSVs.

## GUI

Click **Sapiens2 3D Kinematics**, browse to a Sapiens2 REC3D `.c3d` file,
optionally a `--keypoint-map` JSON, set the Butterworth cutoff (default 6 Hz;
0 disables filtering) and an optional neutral-frame range, then **Run**. The
equivalent CLI command is printed to the terminal before the run starts.
Outputs are written to a timestamped
`processed_sapiens3d_kinematics_YYYYMMDD_HHMMSS/` directory next to the
input file.

## CLI

```bash
# No args -> GUI (same as the button)
uv run vaila/sapiens3d_kinematics.py

# Headless run
uv run vaila/sapiens3d_kinematics.py -i path/to/rec3d.c3d \
    [--keypoint-map map.json] [--cutoff 6.0] \
    [--neutral-frames 0:30] [--output-dir OUTPUT_DIR] [--no-plots]
```

## Outputs

Written to `processed_sapiens3d_kinematics_YYYYMMDD_HHMMSS/`:

- `<stem>_segment_rotations_raw.csv`, `<stem>_joint_rotations_raw.csv` —
  full 3×3 rotation matrices per frame.
- `<stem>_segment_angles_raw.csv`, `<stem>_joint_angles_raw.csv` — quaternion
  (`qx,qy,qz,qw`), both Euler sequences (in degrees), and per-frame QC
  (`det`, `orth_err`) columns.
- `*_neutral.csv` variants of the above (only when `--neutral-frames` is
  given).
- `<stem>_kinematics_metadata.json` — resolved landmark mapping, resolution
  method, quaternion/Euler convention notes, frame definitions, and the
  scientific-limitations text.
- QC plots (unless `--no-plots`): joint Euler angles, SO(3) determinant/
  orthogonality error, quaternion components (left side), pelvis frame
  determinant, and a raw-vs-neutral-zeroed hip flexion comparison.

## Reused vailá utilities

- `vaila.readc3d_export.get_point_labels` — 255-cap-safe C3D point-label
  read (not reimplemented).
- `vaila.filter_utils.butter_filter` — Butterworth zero-phase low-pass
  filtering of the 3D trajectories.
- `scipy.spatial.transform.Rotation` — matrix ↔ quaternion ↔ Euler
  conversions.

`rotation.py` and `joint_kinematics.py` are deliberately **not** reused for
the frame/rotation math itself (different scope and quaternion convention —
see above); their own conventions are unaffected by this module.

## Tests

```bash
uv run pytest tests/test_sapiens3d_kinematics.py -v
```

Covers landmark resolution (direct label match, canonical-308 match,
explicit `--keypoint-map`, and the deliberate refusal when neither applies),
synthetic known-rotation validation (frame properness, SO(3) projection,
Euler round-trips for all three joint sequences, quaternion unit-norm/
scalar-last convention, temporal sign continuity, neutral-zero calibration),
and real-fixture QC against `tests/viewc3d/rec3d_240hz.c3d` (308-point
canonical mapping; confirms `tests/viewc3d/rec3d_200hz.c3d`, 224 points,
correctly raises `LandmarkResolutionError` instead of guessing).
