"""
Project: vailá
Script: joint_kinematics.py
Authors: Paulo Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila

Creation Date: 05 August 2026
Update Date: 05 August 2026
Version: 0.3.99

Description:
    Shared math for turning SAM 3D Body's per-joint rotation output into
    biomechanically usable joint angles (Euler/Cardan degrees + quaternion),
    for ``sam3dinov3.py``/``sam3dinov3_visualize.py`` (monocular camera
    space) and ``rec3d_one_dlt3d.py`` (DLT-triangulated world space, which
    re-exports these unchanged -- see "Why local angles need no realignment"
    below).

    vailá has no single existing 3D joint-angle convention to copy: three
    partial, mutually inconsistent ones exist already --
      - ``rotation.py``: ``scipy`` ``Rotation.from_matrix(...).as_euler("xyz")``,
        but only SEGMENT-vs-LAB-FRAME angles (never one segment relative to
        its parent), used by ``cluster_analysis.py``/``mocap_analysis.py``.
      - ``mpangles.py``: pure 2D vector angles, no rotation matrices at all.
      - IMU modules (``imu_analysis.py``, ``vaila_deadlift_imu.py``): explicit
        scalar-first ``[w, x, y, z]`` quaternions. ``rotation.py.rotmat2quat``
        also *claims* scalar-first in its docstring but actually returns
        scipy's scalar-last ``[x, y, z, w]`` -- a real inconsistency this
        module deliberately does not repeat (see ``rotmat_to_quat_wxyz``).
    This module keeps the Euler sequence ``rotation.py`` already established
    ("xyz", the repo-wide default outside the IMU modules' own aerospace
    ZYX convention) and the IMU modules' scalar-first quaternion convention,
    so a new user reading both kinds of vailá output is not faced with a
    third, unrelated scheme.

    Source of the rotations: SAM 3D Body's MHR (Momentum Human Rig) model
    regresses a full per-joint GLOBAL rotation for its own 127-joint rig
    (``out["pred_global_rots"]``, not the 70-keypoint ``MHR70_NAMES`` list --
    that is a position-only subset for 2D/3D keypoint output). This is a
    real regressed body pose, not a 3-point plane heuristic reconstructed
    from joint positions -- the latter cannot resolve rotation about a
    segment's own long axis (e.g. femur internal/external rotation), which
    a plain hip-knee-ankle plane leaves undetermined.

    The 127-joint kinematic tree (``MHR127_PARENTS``) was extracted directly
    from the shipped ``assets/mhr_model.pt`` TorchScript checkpoint's
    ``character_torch.skeleton.joint_parents`` buffer -- this is Meta's
    "Momentum" rig, loaded upstream from an FBX file via ``pymomentum``
    (a native dependency not part of vailá's own dependency tree and not
    installed here), so the FBX's per-joint NAMES are not available in the
    shipped checkpoint or in the ``sam_3d_body``/``mhr`` Python packages
    (confirmed: neither ships a joint-name list; TorchScript buffers cannot
    hold strings). Names for the ~70 joints that matter for gait/posture
    analysis are instead recovered empirically by matching each rig joint's
    3D position against the already-named ``MHR70_NAMES`` keypoint set for
    the same frame -- see ``infer_joint_names_from_positions``. The
    remaining ~57 rig joints (spine subdivisions, fingers, jaw) are left as
    generic ``joint_NNN`` labels.

    Local (parent-relative) vs. global angles: biomechanical "joint angle"
    (flexion/extension, ab/adduction) means the CHILD segment's rotation
    relative to its PARENT segment, not relative to the camera or lab frame
    -- ``local_rotations_from_global`` computes exactly that from the
    model's per-joint global rotations and the kinematic tree above.

    Why local angles need no realignment for the DLT/world pipeline: a local
    joint rotation is intrinsic to the body's own articulation (an elbow
    bent 90 degrees is 90 degrees no matter which way the camera or the
    world frame is oriented), so it is invariant under any transform that
    only reorients/rescales/translates the RESULT (as ``rec3d_one_dlt3d.py``
    -- ``mesh_alignment.py``'s Umeyama fit -- does to place a monocular
    mesh into DLT-triangulated world space). ``rec3d_one_dlt3d.py`` can
    therefore simply re-export, per solved frame, the winning camera's own
    local joint-angle row -- the same per-frame camera selection its mesh
    alignment already makes -- with no rotation composition of its own.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

#: Parent-joint index per MHR joint, root-first (index 0 = pelvis, parent -1).
#: Extracted from ``vaila/models/sam-3d-dinov3/assets/mhr_model.pt``'s
#: ``character_torch.skeleton.joint_parents`` buffer (Meta's Momentum/MHR
#: rig). A parent index always precedes its child's index (topologically
#: sorted root-to-leaf), so a single forward pass is enough to compute local
#: rotations -- no separate topological sort needed. Kept as a literal so
#: CSV headers and angle math stay available without loading the checkpoint.
MHR127_PARENTS: tuple[int, ...] = (
    -1,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    3,
    3,
    3,
    3,
    2,
    2,
    2,
    2,
    2,
    1,
    18,
    19,
    20,
    21,
    22,
    23,
    19,
    19,
    19,
    19,
    18,
    18,
    18,
    18,
    18,
    1,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    42,
    48,
    49,
    50,
    42,
    52,
    53,
    54,
    42,
    56,
    57,
    58,
    42,
    60,
    61,
    62,
    63,
    40,
    40,
    40,
    40,
    39,
    39,
    39,
    39,
    39,
    37,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    78,
    84,
    85,
    86,
    78,
    88,
    89,
    90,
    78,
    92,
    93,
    94,
    78,
    96,
    97,
    98,
    99,
    76,
    76,
    76,
    76,
    75,
    75,
    75,
    75,
    75,
    37,
    110,
    110,
    110,
    113,
    114,
    114,
    114,
    117,
    118,
    119,
    120,
    113,
    122,
    113,
    124,
    113,
)
MHR127_NUM_JOINTS = len(MHR127_PARENTS)


def local_rotations_from_global(
    global_rots: np.ndarray, parents: tuple[int, ...] = MHR127_PARENTS
) -> np.ndarray:
    """Child-relative-to-parent rotations from the model's own global ones.

    Args:
        global_rots: (J, 3, 3) global (camera/root-frame) rotation matrix
            per joint, e.g. ``instance["global_rots"]``.
        parents: parent joint index per joint, root's parent is -1. Must be
            topologically sorted (parent index < child index), as
            ``MHR127_PARENTS`` is.

    Returns:
        (J, 3, 3) local rotation matrices: ``local[j] = global[parent[j]].T
        @ global[j]`` for non-root joints, ``local[root] = global[root]``
        (the root has no parent to be relative to, so its "local" rotation
        is its own global orientation).
    """
    global_rots = np.asarray(global_rots, dtype=np.float64)
    n = global_rots.shape[0]
    if len(parents) != n:
        raise ValueError(f"parents has {len(parents)} entries but global_rots has {n} joints")
    local = np.empty_like(global_rots)
    for j in range(n):
        p = parents[j]
        if p < 0:
            local[j] = global_rots[j]
        else:
            local[j] = global_rots[p].T @ global_rots[j]
    return local


def rotmat_to_euler_xyz_deg(rotmats: np.ndarray) -> np.ndarray:
    """Cardan/Tait-Bryan XYZ Euler angles in degrees, matching rotation.py.

    Args:
        rotmats: (..., 3, 3) rotation matrices (any leading batch shape,
            e.g. (J, 3, 3) or (N, J, 3, 3)).

    Returns:
        (..., 3) angles in degrees, same leading shape, columns (x, y, z).
    """
    rotmats = np.asarray(rotmats, dtype=np.float64)
    flat = rotmats.reshape(-1, 3, 3)
    euler = Rotation.from_matrix(flat).as_euler("xyz", degrees=True)
    return euler.reshape(*rotmats.shape[:-2], 3)


def rotmat_to_quat_wxyz(rotmats: np.ndarray) -> np.ndarray:
    """Scalar-first (w, x, y, z) quaternions, matching the IMU modules.

    scipy's own ``Rotation.as_quat()`` is scalar-LAST (x, y, z, w); this
    function reorders it to scalar-first so a joint-angle CSV and an IMU CSV
    read the same way in this codebase (unlike ``rotation.py.rotmat2quat``,
    whose docstring claims scalar-first but returns scipy's raw scalar-last
    order -- a real inconsistency this function deliberately avoids).

    Args:
        rotmats: (..., 3, 3) rotation matrices.

    Returns:
        (..., 4) quaternions, columns (w, x, y, z), unit norm.
    """
    rotmats = np.asarray(rotmats, dtype=np.float64)
    flat = rotmats.reshape(-1, 3, 3)
    xyzw = Rotation.from_matrix(flat).as_quat()  # scipy: scalar-last
    wxyz = xyzw[:, [3, 0, 1, 2]]
    return wxyz.reshape(*rotmats.shape[:-2], 4)


def infer_joint_names_from_positions(
    rig_positions: np.ndarray,
    named_positions: np.ndarray,
    named_names: list[str],
    tol_m: float = 0.03,
) -> list[str]:
    """Recover joint names for the 127-joint rig from a 70-name keypoint set.

    Matches each NAMED keypoint (e.g. MHR70's ``left-knee``) to its nearest
    rig joint by 3D position; a rig joint farther than ``tol_m`` from every
    named keypoint keeps a generic ``joint_NNN`` label. Iterating from the
    named side (rather than assigning every rig joint its single nearest
    named point) guarantees at most ``len(named_names)`` assignments, so two
    named keypoints cannot silently collide on the same rig index.

    This is a one-time, per-run computation (the mapping is a fixed model
    property, not something that changes frame to frame) -- call it once on
    any frame with both ``pred_joint_coords`` and ``pred_keypoints_3d``
    populated for the same person, and reuse the result for every frame.

    Args:
        rig_positions: (127, 3) MHR rig joint positions (root-relative
            metres), e.g. ``instance["joint_coords_3d"]``.
        named_positions: (K, 3) positions for the K already-named keypoints,
            in the SAME frame/coordinate convention as ``rig_positions``
            (e.g. ``instance["keypoints_3d"]``, K usually 70).
        named_names: the K names, same order as ``named_positions`` (e.g.
            ``MHR70_NAMES``).
        tol_m: maximum matching distance in metres (default 3 cm -- rig
            joints and MHR70 keypoints for the same anatomical point are
            typically millimetres apart in this model, not centimetres, so
            this is a generous tolerance against a genuine mismatch).

    Returns:
        List of length ``len(rig_positions)``: a name from ``named_names``
        for each matched rig joint, else ``f"joint_{idx:03d}"``.
    """
    rig_positions = np.asarray(rig_positions, dtype=np.float64)
    named_positions = np.asarray(named_positions, dtype=np.float64)
    n_rig = rig_positions.shape[0]
    names = [f"joint_{idx:03d}" for idx in range(n_rig)]
    used_rig_idx: set[int] = set()
    for name, pos in zip(named_names, named_positions, strict=False):
        if not np.isfinite(pos).all():
            continue
        dists = np.linalg.norm(rig_positions - pos[None, :], axis=1)
        order = np.argsort(dists)
        for candidate in order:
            candidate = int(candidate)
            if dists[candidate] > tol_m:
                break
            if candidate not in used_rig_idx:
                names[candidate] = name
                used_rig_idx.add(candidate)
                break
    return names
