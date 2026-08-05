# joint_kinematics

## Module information

| Field | Value |
|-------|--------|
| **Category** | Processing |
| **File** | `vaila/joint_kinematics.py` |
| **Version** | 0.3.99 |
| **Author** | Paulo Santiago |
| **GUI** | No |
| **CLI** | No (library module) |

---

## Description

Shared math turning SAM 3D Body's per-joint rotation output into biomechanically usable joint angles (Euler/Cardan degrees + quaternion), consumed by **sam3dinov3** (writes `*_sam3dinov3_joint_angles.csv`), **sam3dinov3_visualize** (filters it by selected ID), and **rec3d_one_dlt3d** (re-exports the winning camera's rows unchanged into the DLT-triangulated output).

*vailá* had no single existing 3D joint-angle convention to copy — three partial, mutually inconsistent ones already existed:

- **rotation.py**: `scipy` `Rotation.from_matrix(...).as_euler("xyz")`, but only SEGMENT-vs-LAB-FRAME angles (never one segment relative to its parent), used by `cluster_analysis.py`/`mocap_analysis.py`.
- **mpangles.py**: pure 2D vector angles, no rotation matrices at all.
- IMU modules (`imu_analysis.py`, `vaila_deadlift_imu.py`): explicit scalar-first `[w, x, y, z]` quaternions. `rotation.py.rotmat2quat` also *claims* scalar-first in its docstring but actually returns scipy's scalar-last `[x, y, z, w]` — a real inconsistency this module deliberately does not repeat.

This module keeps `rotation.py`'s established Euler sequence (`"xyz"`, the repo-wide default outside the IMU modules' own aerospace ZYX convention) and the IMU modules' scalar-first quaternion convention.

**Source of the rotations:** SAM 3D Body's MHR (Momentum Human Rig) model regresses a full per-joint GLOBAL rotation for its own **127-joint rig** (`out["pred_global_rots"]`, not the 70-keypoint `MHR70_NAMES` list — that is a position-only subset for 2D/3D keypoint output). This is the model's own regressed body pose, not a 3-point plane heuristic reconstructed from joint positions — the latter cannot resolve rotation about a segment's own long axis (e.g. femur internal/external rotation), which a plain hip-knee-ankle plane leaves undetermined.

**Kinematic tree:** `MHR127_PARENTS` (127 parent indices, root=0/parent=-1) was extracted directly from the shipped `assets/mhr_model.pt` TorchScript checkpoint's `character_torch.skeleton.joint_parents` buffer — Meta's "Momentum" rig, loaded upstream from an FBX file via `pymomentum` (a native dependency not part of *vailá*'s own dependency tree). The FBX's per-joint **names** are therefore not available in the shipped checkpoint or in the `sam_3d_body`/`mhr` Python packages (neither ships a joint-name list; TorchScript buffers cannot hold strings). Names for the joints that matter for gait/posture analysis are instead recovered empirically by matching each rig joint's 3D position against the already-named `MHR70_NAMES` keypoint set for the same frame (`infer_joint_names_from_positions`) — validated on real GPU output (2026-08-05): 55/70 MHR70 names matched within 3 cm (everything but face/eye/ear landmarks, which have no rig ROTATION joint of their own). The remaining rig joints keep a generic `joint_NNN` label.

**Local vs. global:** biomechanical "joint angle" means the CHILD segment's rotation relative to its PARENT segment, not relative to the camera or lab frame — `local_rotations_from_global` computes exactly that from the model's global rotations and the kinematic tree above.

**Why the DLT/world pipeline needs no realignment:** a local joint rotation is intrinsic to the body's own articulation (an elbow bent 90° is 90° no matter which way the camera or world frame is oriented), so it is invariant under the Umeyama transform `rec3d_one_dlt3d.py` applies to place a monocular mesh into DLT-triangulated world space. `rec3d_one_dlt3d.py` therefore simply re-exports, per solved frame, the winning camera's own local joint-angle row — the same per-frame camera selection its mesh alignment already makes — with no rotation composition of its own.

**Gimbal lock:** any 3-parameter Euler representation has a mathematical singularity at ±90° on the middle axis; `scipy` warns ("Gimbal lock detected... third angle set to zero") and pins the ambiguous angle rather than guessing — observed on real data. The quaternion columns have no such singularity and should be treated as the lossless representation; the Euler columns are for human readability.

This module never loads SAM3/SAM 3D Body weights and has no CUDA/GPU dependency — pure NumPy/scipy, so it runs anywhere.

---

## Main functions

| Function | Description |
|----------|-------------|
| `local_rotations_from_global(global_rots, parents)` | Child-relative-to-parent rotations from the model's own global ones. |
| `rotmat_to_euler_xyz_deg(rotmats)` | Cardan/Tait-Bryan XYZ Euler angles in degrees, matching `rotation.py`'s sequence. |
| `rotmat_to_quat_wxyz(rotmats)` | Scalar-first `(w, x, y, z)` quaternions, matching the IMU modules (scipy itself is scalar-last). |
| `infer_joint_names_from_positions(rig_positions, named_positions, named_names, tol_m=0.03)` | Recover joint names for the 127-joint rig from a 70-name keypoint set by nearest-position matching. |
| `MHR127_PARENTS` | 127-entry parent-index tuple, the MHR rig's kinematic tree (constant, extracted from the checkpoint). |

---

## Related modules

| Module | Role |
|--------|------|
| **sam3dinov3** | Captures `pred_global_rots`/`pred_joint_coords` from SAM 3D Body and writes `*_sam3dinov3_joint_angles.csv` using this module. |
| **sam3dinov3_visualize** | Filters the joint-angle CSV to the selected person ID, like its other per-person CSVs. |
| **rec3d_one_dlt3d** | Re-exports the winning camera's local joint-angle rows into `<file_base>_joint_angles.csv` alongside the DLT-triangulated mesh/BVH. |
| **rotation.py** | The pre-existing (segment-vs-lab-frame) 3D angle convention this module deliberately stays consistent with (Euler sequence) and deliberately corrects (quaternion scalar order). |

---

## Testing

- `tests/test_joint_kinematics.py` — synthetic, CPU-only: kinematic-tree topology, local-rotation recovery against a known ground-truth delta, Euler/quaternion round-trips, name-inference matching/tie-breaking/NaN-handling.
- `tests/test_sam3dinov3.py` — `_instances_from_outputs`/`write_long_joint_angles_csv`/`_joint_rig_names` with synthetic (but shape/orthonormality-correct) rotation data.
- Smoke-tested manually against a real forward pass on an RTX 4090 (2026-08-05): `pred_global_rots` shape `(127, 3, 3)`, every sampled rotation orthonormal (det ≈ 1.0000), 55/70 names matched, decoded angles biomechanically plausible (knee ~44° flexion, hip ~64°, ankle ~13° for a mid-stride frame). Not repeatable in CI (needs gated CUDA weights).

---

Part of *vailá* - Multimodal Toolbox  
[GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
