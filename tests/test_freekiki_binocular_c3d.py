"""The two-camera C3D of the field points (vaila/freekiki_binocular_c3d.py).

A ground point seen by two synthetic cameras must come back exactly, with z = 0 by construction and a
reprojection residual of zero. A point off the ground is still solved on the plane, and its residual
shows the error, so the 2D gate can reject it.
"""

from __future__ import annotations

import numpy as np

from vaila import freekiki_binocular_c3d as bc
from vaila.fifa_to_dlt import compute_dlt3d_from_KRt
from vaila.freekiki_stereo import project
from vaila.monocular_dlt_align import dlt_projection_matrix

W, H = 1920.0, 1080.0
K = np.array([[2400.0, 0.0, W / 2.0], [0.0, 2400.0, H / 2.0], [0.0, 0.0, 1.0]])


def _camera(centre, target) -> np.ndarray:
    centre, target = np.asarray(centre, float), np.asarray(target, float)
    z = target - centre
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    return compute_dlt3d_from_KRt(K, R, -R @ centre)


L_A = _camera([-60.0, -45.0, 25.0], [-30.0, -10.0, 1.0])
L_B = _camera([-75.0, 20.0, 22.0], [-30.0, -10.0, 1.0])


def _pixel(L: np.ndarray, xyz) -> tuple[float, float]:
    uv = project(dlt_projection_matrix(L), np.asarray(xyz, dtype=float))
    return float(uv[0]), float(uv[1])


def test_a_ground_point_is_recovered_on_the_plane() -> None:
    truth = np.array([-31.2, 4.7, 0.0])
    xyz, res_a, res_b = bc.ground_triangulate(L_A, L_B, _pixel(L_A, truth), _pixel(L_B, truth))
    assert np.allclose(xyz, truth, atol=1e-6)
    assert res_a < 1e-6 and res_b < 1e-6


def test_a_point_above_the_ground_shows_its_error_in_the_residual() -> None:
    truth = np.array([-31.2, 4.7, 1.5])  # not on the field: the plane solution is off
    xyz, res_a, res_b = bc.ground_triangulate(L_A, L_B, _pixel(L_A, truth), _pixel(L_B, truth))
    assert xyz[2] == 0.0
    assert max(res_a, res_b) > bc.MAX_KP_RESIDUAL_PX


def test_the_field_pixels_are_read_per_frame(tmp_path) -> None:
    path = tmp_path / "field_kps_getpixelvideo.csv"
    path.write_text(
        "frame,p0_x,p0_y,p1_x,p1_y\n0,10.5,20.0,,\n1,11.0,21.0,30.0,40.0\n", encoding="utf-8"
    )
    pixels = bc._field_2d(path, 2)
    assert pixels[0] == {0: (10.5, 20.0)}
    assert pixels[1] == {0: (11.0, 21.0), 1: (30.0, 40.0)}
