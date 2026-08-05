"""Synthetic, CPU-only unit tests for vaila.joint_kinematics.

The real-GPU capture path (SAM 3D Body's pred_global_rots/pred_joint_coords
-> vaila.sam3dinov3._instances_from_outputs -> write_long_joint_angles_csv)
was smoke-tested manually against a real forward pass on an RTX 4090
(2026-08-05, frame 231 of c1_cod.mp4): pred_global_rots came back shape
(127, 3, 3), every sampled rotation matrix was orthonormal (det ~1.0000,
||R Rt - I|| ~1e-7), 55/70 MHR70 names matched a rig joint within 3 cm
(everything but face/eye/ear landmarks, which have no rig ROTATION joint of
their own), and decoded joint angles were biomechanically plausible
(knee ~44 deg flexion, hip ~64 deg, ankle ~13 deg for a mid-stride frame).
That path needs real CUDA weights and is not repeatable in CI; these tests
cover the deterministic math it depends on.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

try:
    from vaila.joint_kinematics import (
        MHR127_NUM_JOINTS,
        MHR127_PARENTS,
        infer_joint_names_from_positions,
        local_rotations_from_global,
        rotmat_to_euler_xyz_deg,
        rotmat_to_quat_wxyz,
    )
except ImportError:
    from joint_kinematics import (  # ty: ignore[unresolved-import]
        MHR127_NUM_JOINTS,
        MHR127_PARENTS,
        infer_joint_names_from_positions,
        local_rotations_from_global,
        rotmat_to_euler_xyz_deg,
        rotmat_to_quat_wxyz,
    )


def test_mhr127_parents_is_topologically_sorted():
    """Every joint's parent index must precede it (root=0, parent=-1)."""
    assert len(MHR127_PARENTS) == MHR127_NUM_JOINTS == 127
    assert MHR127_PARENTS[0] == -1
    assert all(MHR127_PARENTS[j] < j for j in range(1, MHR127_NUM_JOINTS))


def test_local_rotations_root_equals_its_own_global():
    rng = np.random.default_rng(0)
    global_rots = Rotation.random(MHR127_NUM_JOINTS, random_state=rng).as_matrix()
    local = local_rotations_from_global(global_rots)
    assert np.allclose(local[0], global_rots[0])


def test_local_rotations_identity_everywhere_stays_identity():
    global_rots = np.stack([np.eye(3)] * MHR127_NUM_JOINTS)
    local = local_rotations_from_global(global_rots)
    assert np.allclose(local, np.eye(3))


def test_local_rotations_recovers_a_known_relative_rotation():
    """local[child] must equal exactly the rotation applied on top of its parent."""
    global_rots = np.stack([np.eye(3)] * MHR127_NUM_JOINTS)
    child = 5
    parent = MHR127_PARENTS[child]
    assert parent >= 0

    parent_orientation = Rotation.from_euler("xyz", [12, -34, 56], degrees=True).as_matrix()
    delta = Rotation.from_euler("xyz", [10, 20, 30], degrees=True).as_matrix()
    global_rots[parent] = parent_orientation
    global_rots[child] = parent_orientation @ delta

    local = local_rotations_from_global(global_rots)
    assert np.allclose(local[child], delta, atol=1e-9)
    # A joint whose parent is untouched (not `parent`, not `child` itself)
    # must be unaffected -- unlike joint 6, whose parent IS joint 5 (`child`
    # here), so its own local rotation would change too and is not a valid
    # "unrelated" example.
    other = next(
        j
        for j in range(1, MHR127_NUM_JOINTS)
        if j not in (parent, child) and MHR127_PARENTS[j] not in (parent, child)
    )
    assert np.allclose(local[other], np.eye(3))


def test_local_rotations_rejects_mismatched_parents_length():
    global_rots = np.stack([np.eye(3)] * 5)
    with pytest.raises(ValueError, match="parents has"):
        local_rotations_from_global(global_rots, parents=(-1, 0, 0))


def test_rotmat_to_euler_xyz_deg_matches_scipy_convention():
    R = Rotation.from_euler("xyz", [15, -25, 40], degrees=True).as_matrix()
    euler = rotmat_to_euler_xyz_deg(R[None])[0]
    assert np.allclose(euler, [15, -25, 40], atol=1e-9)


def test_rotmat_to_euler_xyz_deg_preserves_batch_shape():
    rng = np.random.default_rng(1)
    rotmats = Rotation.random(4 * MHR127_NUM_JOINTS, random_state=rng).as_matrix()
    rotmats = rotmats.reshape(4, MHR127_NUM_JOINTS, 3, 3)
    euler = rotmat_to_euler_xyz_deg(rotmats)
    assert euler.shape == (4, MHR127_NUM_JOINTS, 3)


def test_rotmat_to_quat_wxyz_is_scalar_first():
    """Deliberately the OPPOSITE convention from scipy's own as_quat() (scalar-last),
    matching the IMU modules (imu_analysis.py, vaila_deadlift_imu.py) rather than
    repeating rotation.py.rotmat2quat's docstring/behaviour mismatch (see module
    docstring)."""
    R = Rotation.from_euler("xyz", [0, 0, 90], degrees=True).as_matrix()
    quat_wxyz = rotmat_to_quat_wxyz(R[None])[0]
    # A pure +90 deg rotation about Z: scipy's own (scalar-last) xyzw would be
    # [0, 0, sin(45deg), cos(45deg)] = [0, 0, 0.7071, 0.7071].
    expected_w_last = Rotation.from_euler("xyz", [0, 0, 90], degrees=True).as_quat()
    assert np.allclose(quat_wxyz, expected_w_last[[3, 0, 1, 2]], atol=1e-9)


def test_rotmat_to_quat_wxyz_roundtrips_through_scipy():
    rng = np.random.default_rng(2)
    R = Rotation.random(random_state=rng).as_matrix()
    quat_wxyz = rotmat_to_quat_wxyz(R[None])[0]
    R_back = Rotation.from_quat(quat_wxyz[[1, 2, 3, 0]]).as_matrix()  # back to scipy xyzw
    assert np.allclose(R_back, R, atol=1e-9)


def test_infer_joint_names_from_positions_matches_within_tolerance():
    rng = np.random.default_rng(3)
    rig_positions = rng.normal(scale=5.0, size=(MHR127_NUM_JOINTS, 3))
    rig_positions[10] = [1.0, 2.0, 3.0]
    rig_positions[50] = [4.0, 5.0, 6.0]

    named_positions = np.array([[1.0, 2.0, 3.001], [4.0, 5.0, 6.002], [100.0, 100.0, 100.0]])
    named_names = ["left-knee", "right-elbow", "nose"]

    names = infer_joint_names_from_positions(
        rig_positions, named_positions, named_names, tol_m=0.03
    )
    assert names[10] == "left-knee"
    assert names[50] == "right-elbow"
    # "nose" is far from every rig joint here, so it must match nothing.
    assert "nose" not in names
    # Any unmatched joint keeps a generic label.
    assert names[0] == "joint_000"


def test_infer_joint_names_from_positions_never_double_assigns_a_rig_joint():
    """Two named keypoints that are both closest to the SAME rig joint must not
    both claim it -- the second-closest named point should fall back to its
    next-nearest rig joint instead."""
    rig_positions = np.zeros((MHR127_NUM_JOINTS, 3))
    rig_positions[0] = [0.0, 0.0, 0.0]
    rig_positions[1] = [0.02, 0.0, 0.0]  # slightly farther, still within tolerance
    for i in range(2, MHR127_NUM_JOINTS):
        rig_positions[i] = [100.0 + i, 0.0, 0.0]

    named_positions = np.array([[0.001, 0.0, 0.0], [0.005, 0.0, 0.0]])
    named_names = ["a", "b"]

    names = infer_joint_names_from_positions(
        rig_positions, named_positions, named_names, tol_m=0.03
    )
    assigned = {names[0], names[1]}
    assert assigned == {"a", "b"}


def test_infer_joint_names_from_positions_ignores_nan_named_points():
    rig_positions = np.zeros((MHR127_NUM_JOINTS, 3))
    rig_positions[3] = [1.0, 1.0, 1.0]
    named_positions = np.array([[np.nan, np.nan, np.nan], [1.0, 1.0, 1.0]])
    names = infer_joint_names_from_positions(
        rig_positions, named_positions, ["broken", "good"], tol_m=0.03
    )
    assert names[3] == "good"
