"""Occlusion handling across the rec3d 3D-visualization exports.

Real markerless input (SAM3+Sapiens2 / SAM3+DINOv3) carries NaN wherever a
keypoint was occluded or below confidence. Each of the three export formats
has a different correct answer for what to do with those samples, and each
one used to get it wrong in a way that only showed up as a visual artifact
in Blender:

  * C3D  — has a real invalid-sample convention (negative residual). Writing
    0.0 with a *valid* residual parked every occluded marker on the world
    origin, where viewers draw it as a genuine marker.
  * BVH  — has no such convention, so a gap must be filled; writing 0.0
    teleported the joint to the origin mid-motion.
  * mesh — a frame with a single occluded alignment marker was dropped
    entirely, leaving holes in the OBJ sequence that silently desynchronise
    a Blender OBJ-sequence import from the C3D/BVH.

These tests are synthetic and CPU-only; the real-data tier lives in
tests/test_rec3d_mesh_export.py.
"""

import numpy as np
import pandas as pd
import pytest

try:
    from vaila.mesh_alignment import (
        best_camera_alignment,
        interpolate_similarity_transform,
        slerp_rotation,
    )
    from vaila.rec3d import save_rec3d_as_bvh
except ImportError:  # standalone execution
    from mesh_alignment import (  # ty: ignore[unresolved-import]
        best_camera_alignment,
        interpolate_similarity_transform,
        slerp_rotation,
    )
    from rec3d import save_rec3d_as_bvh  # ty: ignore[unresolved-import]


def _rotation_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _marker_frame(n_frames=8, n_markers=3):
    """Wide rec3d-convention DataFrame: frame, p1_x, p1_y, p1_z, ..."""
    data = {"frame": np.arange(n_frames, dtype=float)}
    for m in range(1, n_markers + 1):
        for axis, base in zip("xyz", (0.0, 10.0, 20.0), strict=True):
            data[f"p{m}_{axis}"] = base + m + np.arange(n_frames, dtype=float)
    return pd.DataFrame(data)


# --------------------------------------------------------------------------
# BVH: occluded samples must be gap-filled, never written as origin spikes
# --------------------------------------------------------------------------


def _read_bvh_motion(path):
    lines = path.read_text().splitlines()
    idx = lines.index("MOTION")
    n_frames = int(lines[idx + 1].split(":")[1])
    rows = [line.split() for line in lines[idx + 3 :] if line.strip()]
    assert len(rows) == n_frames
    return np.array(rows, dtype=float)


def test_bvh_interpolates_interior_occlusion_instead_of_writing_origin(tmp_path):
    df = _marker_frame()
    # Occlude marker p2 in the middle of the trial only.
    df.loc[3:4, ["p2_x", "p2_y", "p2_z"]] = np.nan

    out = save_rec3d_as_bvh(df, str(tmp_path), "take", point_rate=100.0, gui=False, swap_yz=False)
    motion = _read_bvh_motion(tmp_path / "take.bvh")

    # p2 occupies channels 3:6 (markers are emitted in p1, p2, p3 order).
    p2 = motion[:, 3:6]
    assert out is not None
    # The gap must not collapse to the origin...
    assert not np.allclose(p2[3], 0.0), "occluded sample written as origin spike"
    assert not np.allclose(p2[4], 0.0)
    # ...and must sit between the bracketing valid samples.
    assert np.all(p2[3] > p2[2]) and np.all(p2[3] < p2[5])
    assert np.all(p2[4] > p2[3]) and np.all(p2[4] < p2[5])


def test_bvh_holds_edge_samples_when_gap_touches_trial_start(tmp_path):
    df = _marker_frame()
    df.loc[0:1, ["p1_x", "p1_y", "p1_z"]] = np.nan

    save_rec3d_as_bvh(df, str(tmp_path), "take", point_rate=100.0, gui=False, swap_yz=False)
    motion = _read_bvh_motion(tmp_path / "take.bvh")

    p1 = motion[:, 0:3]
    assert not np.allclose(p1[0], 0.0)
    # Leading gap holds the first valid sample (frame 2).
    np.testing.assert_allclose(p1[0], p1[2])
    np.testing.assert_allclose(p1[1], p1[2])


def test_bvh_leaves_never_seen_marker_at_origin_without_crashing(tmp_path):
    df = _marker_frame()
    df[["p3_x", "p3_y", "p3_z"]] = np.nan

    save_rec3d_as_bvh(df, str(tmp_path), "take", point_rate=100.0, gui=False, swap_yz=False)
    motion = _read_bvh_motion(tmp_path / "take.bvh")

    assert np.allclose(motion[:, 6:9], 0.0)
    assert np.isfinite(motion).all(), "BVH must never contain NaN/inf"


def test_bvh_without_occlusion_is_unchanged_by_gap_filling(tmp_path):
    df = _marker_frame()
    save_rec3d_as_bvh(df, str(tmp_path), "take", point_rate=100.0, gui=False, swap_yz=False)
    motion = _read_bvh_motion(tmp_path / "take.bvh")

    expected = np.column_stack([df[f"p{m}_{axis}"].to_numpy() for m in (1, 2, 3) for axis in "xyz"])
    np.testing.assert_allclose(motion, expected, atol=1e-6)


# --------------------------------------------------------------------------
# Mesh alignment: a partially occluded frame is solvable, not droppable
# --------------------------------------------------------------------------


def test_alignment_survives_occlusion_of_some_markers():
    """A frame keeps a real fit when only some alignment markers are NaN.

    This is the condition that used to drop frames out of the mesh sequence:
    on the real fixture only the two acromions ever go missing, leaving seven
    perfectly good correspondences behind.
    """
    rng = np.random.default_rng(0)
    source = rng.normal(size=(9, 3))
    R, s, t = _rotation_z(0.7), 1.4, np.array([2.0, -3.0, 0.5])
    target = s * (source @ R.T) + t

    occluded_target = target.copy()
    occluded_target[[7, 8]] = np.nan  # the "acromion" rows

    idx, result = best_camera_alignment([source], occluded_target)

    assert idx == 0
    assert result is not None and not result.degenerate
    assert result.n_points == 7, "occluded rows must be dropped, not the frame"
    np.testing.assert_allclose(result.R, R, atol=1e-8)
    assert result.s == pytest.approx(s, abs=1e-8)
    np.testing.assert_allclose(result.t, t, atol=1e-8)
    assert result.mean_residual < 1e-8


def test_alignment_rejects_frame_when_too_few_markers_remain():
    rng = np.random.default_rng(1)
    source = rng.normal(size=(9, 3))
    target = 1.1 * source + 0.5
    target[3:] = np.nan  # only 3 correspondences left, below min_points=4

    idx, result = best_camera_alignment([source], target)
    assert idx is None and result is None


# --------------------------------------------------------------------------
# Transform interpolation (the fallback used only when a frame is unsolvable)
# --------------------------------------------------------------------------


def test_slerp_rotation_endpoints_and_midpoint_stay_on_so3():
    R0, R1 = _rotation_z(0.0), _rotation_z(np.pi / 2)

    np.testing.assert_allclose(slerp_rotation(R0, R1, 0.0), R0, atol=1e-12)
    np.testing.assert_allclose(slerp_rotation(R0, R1, 1.0), R1, atol=1e-12)

    mid = slerp_rotation(R0, R1, 0.5)
    np.testing.assert_allclose(mid, _rotation_z(np.pi / 4), atol=1e-12)
    # Still a proper rotation, which a component-wise blend would not be.
    np.testing.assert_allclose(mid @ mid.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(mid) == pytest.approx(1.0, abs=1e-12)


def test_interpolate_similarity_transform_blends_between_neighbours():
    before = (10.0, _rotation_z(0.0), 1.0, np.array([0.0, 0.0, 0.0]))
    after = (20.0, _rotation_z(np.pi / 2), 2.0, np.array([10.0, 0.0, 0.0]))

    R, s, t = interpolate_similarity_transform(before, after, 15.0)

    np.testing.assert_allclose(R, _rotation_z(np.pi / 4), atol=1e-12)
    assert s == pytest.approx(1.5)
    np.testing.assert_allclose(t, [5.0, 0.0, 0.0], atol=1e-12)


def test_interpolate_similarity_transform_holds_single_sided_neighbour():
    known = (10.0, _rotation_z(0.3), 1.7, np.array([1.0, 2.0, 3.0]))

    for before, after in ((known, None), (None, known)):
        R, s, t = interpolate_similarity_transform(before, after, 99.0)
        np.testing.assert_allclose(R, known[1], atol=1e-12)
        assert s == pytest.approx(known[2])
        np.testing.assert_allclose(t, known[3], atol=1e-12)


def test_interpolate_similarity_transform_returns_none_without_neighbours():
    assert interpolate_similarity_transform(None, None, 5.0) is None


# --------------------------------------------------------------------------
# Blender axis convention: the Z-up conversion must not mirror the subject
# --------------------------------------------------------------------------


def _blender_world_from_bvh(xyz):
    """Blender's BVH importer maps a Y-up file (X, Y, Z) to world (X, -Z, Y)."""
    x, y, z = xyz
    return np.array([x, -z, y])


def test_bvh_swap_yz_round_trips_to_the_original_dlt_coordinates(tmp_path):
    """A DLT point must land back on itself in the Blender scene.

    save_rec3d_as_bvh(swap_yz=True) writes (x, z, -y); Blender's importer
    rotates that back, so the marker ends up at the original (x, y, z) and
    therefore on top of the same marker imported from the C3D.
    """
    df = _marker_frame(n_frames=4, n_markers=2)
    save_rec3d_as_bvh(df, str(tmp_path), "take", point_rate=100.0, gui=False, swap_yz=True)
    motion = _read_bvh_motion(tmp_path / "take.bvh")

    for m, offset in ((1, 0), (2, 3)):
        for frame in range(4):
            want = np.array([df.loc[frame, f"p{m}_{a}"] for a in "xyz"])
            got = _blender_world_from_bvh(motion[frame, offset : offset + 3])
            np.testing.assert_allclose(got, want, atol=1e-6)


def test_bvh_swap_yz_preserves_anatomical_left_and_right(tmp_path):
    """The exported animation must not be mirrored.

    Writing (x, z, y) is a reflection; combined with the importer's own
    rotation it swaps the subject's left and right in the scene, which
    inverts any asymmetry (push-off limb, limb dominance) read off it.
    """
    # Subject at the origin: up = +z, facing = +x, so anatomical left = +y.
    df = pd.DataFrame(
        {
            "frame": [0.0],
            "p1_x": [0.0],
            "p1_y": [0.2],
            "p1_z": [1.4],  # left shoulder
            "p2_x": [0.0],
            "p2_y": [-0.2],
            "p2_z": [1.4],  # right shoulder
            "p3_x": [0.0],
            "p3_y": [0.0],
            "p3_z": [0.9],  # mid hip
            "p4_x": [0.3],
            "p4_y": [0.0],
            "p4_z": [1.5],  # nose (facing +x)
        }
    )
    save_rec3d_as_bvh(df, str(tmp_path), "take", point_rate=100.0, gui=False, swap_yz=True)
    motion = _read_bvh_motion(tmp_path / "take.bvh")

    left_sh, right_sh, hip, nose = (
        _blender_world_from_bvh(motion[0, i : i + 3]) for i in (0, 3, 6, 9)
    )

    up = (left_sh + right_sh) / 2 - hip
    facing = nose - hip
    up /= np.linalg.norm(up)
    facing /= np.linalg.norm(facing)
    # In a right-handed frame the left direction is up x facing.
    predicted_left = np.cross(up, facing)
    actual_left = left_sh - right_sh

    assert np.dot(predicted_left, actual_left) > 0, "subject is mirrored in Blender space"
