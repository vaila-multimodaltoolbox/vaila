"""Pure-logic tests for the FreeKiki pitch-frame outputs (no GPU, no weights).

Covers the DLT3D projection and its inverse on a plane, the field keypoints and the
ball in pitch metres, the pose transform (camera -> field, feet on the ground under
the box), and the C3D round trip. The synthetic cameras use the same DLT builder as
the rest of the test suite.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vaila import freekiki_export as export
from vaila import freekiki_field3d as f3
from vaila.fifa_to_dlt import compute_dlt3d_from_KRt

W, H_IMG = 1920, 1080


def _camera(centre: np.ndarray, target: np.ndarray, f: float = 2400.0):
    """K, R (world -> camera, OpenCV axes) and t for a camera at ``centre`` looking at ``target``."""
    up = np.array([0.0, 0.0, 1.0])
    z = target - centre
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    t = -R @ centre
    K = np.array([[f, 0.0, W / 2.0], [0.0, f, H_IMG / 2.0], [0.0, 0.0, 1.0]])
    return K, R, t


def _dlt(centre=(0.0, -60.0, 25.0), target=(0.0, 0.0, 0.0)) -> np.ndarray:
    K, R, t = _camera(np.asarray(centre, float), np.asarray(target, float))
    return compute_dlt3d_from_KRt(K, R, t)


def test_project_then_invert_on_a_plane_recovers_the_ground_point() -> None:
    L = _dlt()
    ground = np.array([[10.0, 5.0, 0.0], [-20.0, -8.0, 0.0], [30.0, 12.0, 0.0]])
    uv = f3.project_points(L, ground)
    for (x, y, z), pix in zip(ground, uv, strict=True):
        got = f3.pixel_to_plane(L, pix, z)
        assert got is not None
        assert got == pytest.approx((x, y), abs=1e-6)


def test_inverse_at_ball_height_recovers_the_centre() -> None:
    # A ball centre at height r: the pixel of the centre inverts to the same x, y.
    L = _dlt()
    centre = np.array([[15.0, -3.0, 0.11]])
    pix = f3.project_points(L, centre)[0]
    got = f3.pixel_to_plane(L, pix, 0.11)
    assert got == pytest.approx((15.0, -3.0), abs=1e-6)


def test_project_marks_points_behind_the_camera_as_nan() -> None:
    L = _dlt()
    behind = np.array([[0.0, -200.0, 0.0]])  # behind the camera at y = -60, looking at +y
    assert np.isnan(f3.project_points(L, behind)).all()


def test_field_keypoints_error_is_the_pixel_distance_to_the_detection() -> None:
    L = _dlt()
    xyz = np.array([[0.0, 0.0, 0.0], [52.45, 33.95, 0.0]])
    names = ["centre", "corner"]
    uv = f3.project_points(L, xyz)
    detected = {0: uv + np.array([3.0, 4.0])}  # 5 px off on both points
    rows = f3.field_keypoints_3d(names, xyz, {0: L}, detected)
    assert [r["name"] for r in rows] == names
    assert [r["err_px"] for r in rows] == pytest.approx([5.0, 5.0])
    assert rows[0]["x_m"] == 0.0 and rows[1]["y_m"] == 33.95


def test_field_keypoints_without_detection_have_nan_error() -> None:
    L = _dlt()
    xyz = np.array([[0.0, 0.0, 0.0]])
    rows = f3.field_keypoints_3d(["centre"], xyz, {7: L}, {})
    assert rows[0]["frame"] == 7
    assert np.isnan(rows[0]["err_px"])


def test_ball_on_the_pitch_and_off_it() -> None:
    L = _dlt()
    inside = f3.project_points(L, np.array([[20.0, 10.0, f3.BALL_RADIUS_M]]))[0]
    outside = f3.project_points(L, np.array([[60.0, 0.0, f3.BALL_RADIUS_M]]))[0]
    rows = [
        {"frame": 0, "cx": inside[0], "cy": inside[1], "score": 0.9},
        {"frame": 1, "cx": outside[0], "cy": outside[1], "score": 0.8},
        {"frame": 2, "cx": "", "cy": "", "score": ""},
    ]
    out = f3.ball_3d(rows, {0: L, 1: L, 2: L})
    assert out[0]["on_field"] is True
    assert out[0]["x_m"] == pytest.approx(20.0, abs=1e-6)
    assert out[0]["z_m"] == f3.BALL_RADIUS_M
    assert out[1]["on_field"] is False
    assert out[2]["on_field"] is False and np.isnan(out[2]["x_m"])


def test_ball_without_a_camera_keeps_nan_coordinates() -> None:
    rows = [{"frame": 5, "cx": 100.0, "cy": 200.0, "score": 0.5}]
    out = f3.ball_3d(rows, {})
    assert np.isnan(out[0]["x_m"]) and out[0]["on_field"] is False


def test_pose_is_rotated_into_the_field_and_its_feet_land_under_the_box() -> None:
    # A person standing at (5, 2) on the pitch, 1.8 m tall. The SAM camera points are
    # the same points seen from the DLT camera (xcam = R (X - C)), so the transform
    # must return the original field coordinates.
    K, R, t = _camera(np.array([0.0, -60.0, 25.0]), np.zeros(3))
    C = -R.T @ t
    L = compute_dlt3d_from_KRt(K, R, t)
    names = ["left-ankle", "right-ankle", "nose"]
    feet_world = np.array([[5.0, 2.0, 0.0], [5.2, 2.0, 0.0]])
    nose_world = np.array([[5.1, 2.0, 1.8]])
    world = np.vstack([feet_world, nose_world])
    xcam = (R @ (world - C).T).T

    feet_px = f3.project_points(L, feet_world).mean(axis=0)
    box = np.array([feet_px[0] - 15, feet_px[1] - 90, feet_px[0] + 15, feet_px[1]])
    pose = f3.pose_to_pitch(names, xcam, L, box)
    assert pose is not None
    # Feet are on the ground (z = 0) under the box base, the nose keeps its height.
    assert pose[0, 2] == pytest.approx(0.0, abs=1e-6)
    assert pose[2, 2] == pytest.approx(1.8, abs=1e-3)
    assert pose[0, :2] == pytest.approx(world[0, :2], abs=1e-6)


def test_pose_without_feet_has_no_anchor() -> None:
    L = _dlt()
    xcam = np.array([[0.0, 0.0, 30.0]])
    assert f3.pose_to_pitch(["nose"], xcam, L, np.array([0.0, 0.0, 10.0, 10.0])) is None


def test_dlt_for_frames_skips_rows_with_nan(tmp_path: Path) -> None:
    L = _dlt()
    path = tmp_path / "cam.dlt3d"
    lines = ["frame," + ",".join(f"L{i}" for i in range(1, 12))]
    lines.append("0," + ",".join(f"{v:.12g}" for v in L))
    lines.append("1," + ",".join(["nan"] * 11))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cams = f3.dlt_for_frames(path)
    assert set(cams) == {0}
    assert f3.dlt_for_frames(None) == {}


def test_write_table_leaves_nan_as_empty_cells(tmp_path: Path) -> None:
    rows = [{"frame": 0, "x_m": float("nan"), "on_field": True}]
    path = f3.write_table(tmp_path / "t.csv", ["frame", "x_m", "on_field"], rows)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines == ["frame,x_m,on_field", "0,,True"]


def test_c3d_round_trip_keeps_points_and_marks_missing_samples(tmp_path: Path) -> None:
    ezc3d = pytest.importorskip("ezc3d")
    points = np.zeros((4, 3, 3))
    points[0, 0] = [1.0, 2.0, 3.0]
    points[1, 0] = [4.0, 5.0, 6.0]
    points[2, 0] = [0.5, 0.25, 0.125]
    points[:, 1] = np.nan  # marker 1 is missing in every frame
    points[1, 2] = np.nan  # marker 2 is missing in frame 1 only
    labels = ["A", "B", "C"]
    path = export.write_c3d(tmp_path / "out.c3d", labels, points, rate_hz=50.0)

    back = ezc3d.c3d(str(path))
    assert back["parameters"]["POINT"]["LABELS"]["value"] == labels
    assert back["parameters"]["POINT"]["RATE"]["value"] == [50.0]
    got = np.transpose(back["data"]["points"][:3], (2, 1, 0))  # (frames, markers, 3)
    assert got[0, 0] == pytest.approx([1.0, 2.0, 3.0], abs=1e-6)
    assert np.isnan(got[:, 1]).all()
    assert np.isnan(got[1, 2]).all() and np.isfinite(got[0, 2]).all()
    # A missing sample is NaN with residual -1 (C3D's invalid mark), so viewers draw nothing.
    residuals = back["data"]["meta_points"]["residuals"][0]  # (markers, frames)
    assert (residuals[1] == -1).all() and residuals[0, 0] == 0


def test_c3d_rejects_bad_shapes_and_duplicate_labels(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        export.write_c3d(tmp_path / "a.c3d", ["A"], np.zeros((2, 2, 3)), rate_hz=25.0)
    with pytest.raises(ValueError):
        export.write_c3d(tmp_path / "b.c3d", ["A", "A"], np.zeros((2, 2, 3)), rate_hz=25.0)
