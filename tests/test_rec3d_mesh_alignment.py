"""Synthetic, CPU-only unit tests for vaila.mesh_alignment.

These tests never touch the real rec3d_todo fixture (see
tests/test_rec3d_mesh_export.py for the real-data regression tier) — they
only prove the Umeyama similarity-fit math and the OBJ I/O helpers are
correct in isolation, using a fixed, known ground-truth transform.
"""

import numpy as np
import pytest

try:
    from vaila.mesh_alignment import (
        AlignmentResult,
        apply_blender_yz_swap,
        apply_similarity_transform,
        best_camera_alignment,
        read_obj_vertices,
        umeyama_alignment,
        write_obj_mesh,
        write_ply_mesh,
    )
except ImportError:
    from mesh_alignment import (  # ty: ignore[unresolved-import]
        AlignmentResult,
        apply_blender_yz_swap,
        apply_similarity_transform,
        best_camera_alignment,
        read_obj_vertices,
        umeyama_alignment,
        write_obj_mesh,
        write_ply_mesh,
    )


def _rotation_matrix(rx, ry, rz):
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


# 10 non-coplanar points: cube vertices plus two off-cube points for spread.
_NON_COPLANAR_POINTS = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.5, 0.5, 2.0],
        [2.0, 0.3, 0.7],
    ],
    dtype=np.float64,
)

_TRUE_R = _rotation_matrix(0.3, -0.5, 1.1)
_TRUE_S = 1.7
_TRUE_T = np.array([1.0, 2.0, -3.0])


def test_umeyama_recovers_known_transform():
    target = apply_similarity_transform(_NON_COPLANAR_POINTS, _TRUE_R, _TRUE_S, _TRUE_T)
    result = umeyama_alignment(_NON_COPLANAR_POINTS, target)

    assert not result.degenerate
    assert result.n_points == len(_NON_COPLANAR_POINTS)
    np.testing.assert_allclose(result.R, _TRUE_R, atol=1e-6)
    assert result.s == pytest.approx(_TRUE_S, abs=1e-6)
    np.testing.assert_allclose(result.t, _TRUE_T, atol=1e-6)
    assert result.mean_residual < 1e-8
    assert result.rms_residual < 1e-8
    assert result.max_residual < 1e-8


def test_umeyama_recovers_transform_under_small_noise():
    rng = np.random.default_rng(42)
    target = apply_similarity_transform(_NON_COPLANAR_POINTS, _TRUE_R, _TRUE_S, _TRUE_T)
    target_noisy = target + rng.normal(scale=1e-3, size=target.shape)
    result = umeyama_alignment(_NON_COPLANAR_POINTS, target_noisy)

    assert not result.degenerate
    np.testing.assert_allclose(result.R, _TRUE_R, atol=5e-3)
    assert result.s == pytest.approx(_TRUE_S, abs=5e-3)
    assert result.mean_residual < 5e-3


def test_umeyama_flags_planar_input_as_degenerate():
    planar_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.5, 0.5, 0.0],
            [2.0, 0.2, 0.0],
        ]
    )
    target = apply_similarity_transform(planar_points, _TRUE_R, _TRUE_S, _TRUE_T)
    result = umeyama_alignment(planar_points, target)

    assert result.degenerate
    assert "planar" in result.reason.lower() or "collinear" in result.reason.lower()
    assert result.R is None


def test_umeyama_flags_too_few_points_as_degenerate():
    result = umeyama_alignment(_NON_COPLANAR_POINTS[:3], _NON_COPLANAR_POINTS[:3])
    assert result.degenerate
    assert "3" in result.reason


def test_umeyama_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        umeyama_alignment(_NON_COPLANAR_POINTS, _NON_COPLANAR_POINTS[:-1])


def test_apply_similarity_transform_on_tetrahedron():
    tetrahedron = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    transformed = apply_similarity_transform(tetrahedron, _TRUE_R, _TRUE_S, _TRUE_T)
    expected = (_TRUE_S * (tetrahedron @ _TRUE_R.T)) + _TRUE_T
    np.testing.assert_allclose(transformed, expected)
    # Rigid+scale transform must preserve inter-vertex distances up to scale.
    original_edge = np.linalg.norm(tetrahedron[1] - tetrahedron[0])
    transformed_edge = np.linalg.norm(transformed[1] - transformed[0])
    assert transformed_edge == pytest.approx(_TRUE_S * original_edge, abs=1e-9)


def test_apply_blender_yz_swap_rotates_vertices_to_y_up():
    vertices = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    faces = np.array([[0, 1, 0]])
    swapped_vertices, _ = apply_blender_yz_swap(vertices, faces)
    # (x, y, z) -> (x, z, -y), NOT the mirroring (x, z, y).
    np.testing.assert_allclose(swapped_vertices, [[1.0, 3.0, -2.0], [4.0, 6.0, -5.0]])


def test_apply_blender_yz_swap_is_a_proper_rotation_not_a_mirror():
    """The conversion must preserve handedness.

    (x, y, z) -> (x, z, y) is a reflection (determinant -1). Blender's BVH
    importer then applies its own Y-up -> Z-up rotation, so a reflected file
    leaves the subject mirrored in the scene: anatomical left and right
    swapped, silently inverting every asymmetry conclusion.
    """
    basis = np.eye(3)
    rotated, _ = apply_blender_yz_swap(basis, np.array([[0, 1, 2]]))
    matrix = rotated.T  # columns are the images of the basis vectors
    assert np.linalg.det(matrix) == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-12)


def test_apply_blender_yz_swap_round_trips_through_blender_import():
    """Blender's importers map a Y-up file (X, Y, Z) to world (X, -Z, Y).

    Composing that with our conversion must return the original DLT point,
    which is what lands the BVH skeleton on the C3D markers in the scene.
    The OBJ mesh does NOT go through this helper -- it is written raw, see
    apply_blender_yz_swap's docstring and test_rec3d_mesh_export.py.
    """
    rng = np.random.default_rng(3)
    pts = rng.normal(size=(20, 3))
    converted, _ = apply_blender_yz_swap(pts, np.array([[0, 1, 2]]))
    in_blender = np.column_stack([converted[:, 0], -converted[:, 2], converted[:, 1]])
    np.testing.assert_allclose(in_blender, pts, atol=1e-12)


def test_apply_blender_yz_swap_keeps_face_winding():
    vertices = np.zeros((4, 3))
    faces = np.array([[0, 1, 2], [1, 2, 3]])
    _, swapped_faces = apply_blender_yz_swap(vertices, faces)
    # A proper rotation preserves orientation, so winding must NOT be
    # reversed -- doing so would turn the mesh inside-out.
    np.testing.assert_array_equal(swapped_faces, [[0, 1, 2], [1, 2, 3]])


def _face_normal_dot_centroid_offset(vertices, face):
    v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
    normal = np.cross(v1 - v0, v2 - v0)
    face_centroid = (v0 + v1 + v2) / 3
    mesh_centroid = vertices.mean(axis=0)
    return float(np.dot(normal, face_centroid - mesh_centroid))


def test_apply_blender_yz_swap_preserves_outward_facing_normals():
    """The Z-up conversion must leave outward normals pointing outward.

    Whichever way each face's normal pointed relative to the mesh centroid
    before the conversion, it must point the same way after -- otherwise the
    mesh renders inside-out in Blender. A proper rotation gives this for
    free; the old reflection-plus-reversed-winding pair also happened to,
    but only by cancelling one error with another.
    """
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    faces = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]])
    original_signs = [np.sign(_face_normal_dot_centroid_offset(vertices, f)) for f in faces]
    assert all(s != 0 for s in original_signs)

    swapped_vertices, swapped_faces = apply_blender_yz_swap(vertices, faces)
    swapped_signs = [
        np.sign(_face_normal_dot_centroid_offset(swapped_vertices, f)) for f in swapped_faces
    ]

    assert swapped_signs == original_signs


def test_best_camera_alignment_picks_lowest_residual_camera():
    target = apply_similarity_transform(_NON_COPLANAR_POINTS, _TRUE_R, _TRUE_S, _TRUE_T)
    good_camera = _NON_COPLANAR_POINTS.copy()
    rng = np.random.default_rng(7)
    noisy_camera = _NON_COPLANAR_POINTS + rng.normal(scale=0.2, size=_NON_COPLANAR_POINTS.shape)

    best_idx, best_result = best_camera_alignment([noisy_camera, good_camera], target)

    assert best_idx == 1
    assert best_result is not None
    assert not best_result.degenerate
    assert best_result.mean_residual < 1e-6


def test_best_camera_alignment_skips_none_and_degenerate_cameras():
    target = apply_similarity_transform(_NON_COPLANAR_POINTS, _TRUE_R, _TRUE_S, _TRUE_T)
    planar_camera = _NON_COPLANAR_POINTS.copy()
    planar_camera[:, 2] = 0.0  # flatten -> degenerate
    good_camera = _NON_COPLANAR_POINTS.copy()

    best_idx, best_result = best_camera_alignment([None, planar_camera, good_camera], target)

    assert best_idx == 2
    assert best_result is not None


def test_best_camera_alignment_returns_none_when_all_unusable():
    target = apply_similarity_transform(_NON_COPLANAR_POINTS, _TRUE_R, _TRUE_S, _TRUE_T)
    best_idx, best_result = best_camera_alignment([None, None], target)
    assert best_idx is None
    assert best_result is None


def test_obj_write_read_round_trip(tmp_path):
    vertices = _NON_COPLANAR_POINTS
    faces = np.array([[0, 1, 2], [0, 2, 3], [1, 2, 3]])
    obj_path = tmp_path / "frame_000000.obj"
    write_obj_mesh(obj_path, vertices, faces)

    read_back = read_obj_vertices(obj_path)
    np.testing.assert_allclose(read_back, vertices, atol=1e-6)

    content = obj_path.read_text()
    assert content.count("\nv ") + (1 if content.startswith("v ") else 0) >= 0
    assert "f 1 2 3" in content  # 0-indexed faces written as 1-indexed


def test_ply_write_round_trip(tmp_path):
    vertices = _NON_COPLANAR_POINTS[:4]
    faces = np.array([[0, 1, 2], [0, 1, 3]])
    ply_path = tmp_path / "frame_000000.ply"
    write_ply_mesh(ply_path, vertices, faces)

    content = ply_path.read_text()
    assert content.startswith("ply\n")
    assert f"element vertex {len(vertices)}" in content
    assert f"element face {len(faces)}" in content


def test_alignment_result_is_a_dataclass_with_expected_fields():
    result = AlignmentResult(degenerate=True, reason="test")
    assert result.R is None
    assert result.degenerate is True
