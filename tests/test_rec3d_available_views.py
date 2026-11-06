"""Triangulation from the cameras that see a point (rec3d.rec3d_available_views).

A NaN in one camera used to discard the whole marker even when two or more other cameras saw
it. These tests pin the new behavior: skip only the missing view, need at least MIN_VIEWS
cameras, and record how many cameras were used (``*_nviews.csv``).
"""

import numpy as np
import pandas as pd

from vaila.rec3d import MIN_VIEWS, rec3d_available_views, rec3d_multicam
from vaila.rec3d_one_dlt3d import run_reconstruction

# Affine "cameras" (L9 = L10 = L11 = 0): u, v are linear in X, Y, Z.
CAM_XY = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=float)  # sees (X, Y)
CAM_YZ = np.array([0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=float)  # sees (Y, Z)
CAM_XZ = np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=float)  # sees (X, Z)
POINT = np.array([5.0, 10.0, 15.0])


def _project(L, X):
    return (
        L[0] * X[0] + L[1] * X[1] + L[2] * X[2] + L[3],
        L[4] * X[0] + L[5] * X[1] + L[6] * X[2] + L[7],
    )


def test_all_views_matches_rec3d_multicam():
    dlts = [CAM_XY, CAM_YZ, CAM_XZ]
    pix = [_project(L, POINT) for L in dlts]
    xyz, used = rec3d_available_views(dlts, pix)
    assert used == 3
    assert np.allclose(xyz, rec3d_multicam(dlts, pix))
    assert np.allclose(xyz, POINT)


def test_missing_view_is_skipped_not_fatal():
    dlts = [CAM_XY, CAM_YZ, CAM_XZ]
    pix = [_project(CAM_XY, POINT), (np.nan, np.nan), _project(CAM_XZ, POINT)]
    xyz, used = rec3d_available_views(dlts, pix)
    assert used == 2
    assert np.allclose(xyz, POINT)


def test_camera_without_dlt_for_the_frame_is_skipped():
    dlts = [CAM_XY, None, CAM_XZ]
    pix = [_project(CAM_XY, POINT), (1.0, 2.0), _project(CAM_XZ, POINT)]
    xyz, used = rec3d_available_views(dlts, pix)
    assert used == 2
    assert np.allclose(xyz, POINT)


def test_fewer_than_min_views_returns_nan():
    assert MIN_VIEWS == 2
    dlts = [CAM_XY, CAM_YZ, CAM_XZ]
    pix = [_project(CAM_XY, POINT), (np.nan, 3.0), (np.inf, np.nan)]
    xyz, used = rec3d_available_views(dlts, pix)
    assert used == 1
    assert np.isnan(xyz).all()


def test_weights_none_is_the_old_unweighted_solution():
    dlts = [CAM_XY, CAM_YZ, CAM_XZ]
    pix = [(5.0, 10.0), (10.4, 15.0), (5.0, 14.6)]  # small inconsistent noise
    assert np.allclose(rec3d_multicam(dlts, pix), rec3d_multicam(dlts, pix, weights=None))
    assert np.allclose(
        rec3d_multicam(dlts, pix), rec3d_multicam(dlts, pix, weights=[1.0, 1.0, 1.0])
    )


def test_low_weight_reduces_the_pull_of_a_bad_view():
    dlts = [CAM_XY, CAM_YZ, CAM_XZ]
    bad = (10.0 + 4.0, 15.0 + 4.0)  # camera YZ is off by 4 units in both axes
    pix = [_project(CAM_XY, POINT), bad, _project(CAM_XZ, POINT)]
    plain = np.linalg.norm(rec3d_multicam(dlts, pix) - POINT)
    weighted = np.linalg.norm(rec3d_multicam(dlts, pix, weights=[1.0, 0.05, 1.0]) - POINT)
    assert weighted < plain / 3


def test_zero_or_nan_weight_drops_the_view():
    dlts = [CAM_XY, CAM_YZ, CAM_XZ]
    pix = [_project(CAM_XY, POINT), (99.0, 99.0), _project(CAM_XZ, POINT)]
    for w in (0.0, np.nan):
        xyz, used = rec3d_available_views(dlts, pix, weights=[1.0, w, 1.0])
        assert used == 2
        assert np.allclose(xyz, POINT)


def _write_csv(path, header, rows):
    lines = [",".join(header)] + [",".join(str(v) for v in row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_cameras(tmp_path, pixel_rows_per_camera):
    """One fixed .dlt3d + one pixel CSV per camera; returns (dlt_files, pixel_files)."""
    dlt_files, pixel_files = [], []
    for i, (L, rows) in enumerate(
        zip([CAM_XY, CAM_YZ, CAM_XZ], pixel_rows_per_camera, strict=True)
    ):
        dlt_path = tmp_path / f"cam{i}.dlt3d"
        _write_csv(dlt_path, ["frame"] + [f"L{k}" for k in range(1, 12)], [[0, *L]])
        pix_path = tmp_path / f"cam{i}.csv"
        _write_csv(pix_path, ["frame", "p0_x", "p0_y", "p1_x", "p1_y"], rows)
        dlt_files.append(str(dlt_path))
        pixel_files.append(str(pix_path))
    return dlt_files, pixel_files


def _run(tmp_path, dlt_files, pixel_files):
    out = tmp_path / "out"
    out.mkdir()
    result = run_reconstruction(dlt_files, pixel_files, str(out), point_rate=100, gui=False)
    assert result is not None
    new_dir, file_base = result
    rec = pd.read_csv(f"{new_dir}/{file_base}.csv")
    nviews = pd.read_csv(f"{new_dir}/{file_base}_nviews.csv")
    return rec, nviews


def test_run_reconstruction_uses_remaining_cameras(tmp_path):
    """End to end: marker p1 is NaN in camera 2 but seen by cameras 1 and 3."""
    p0 = np.array([1.0, 2.0, 3.0])
    per_camera = []
    for i, L in enumerate([CAM_XY, CAM_YZ, CAM_XZ]):
        u1, v1 = (np.nan, np.nan) if i == 1 else _project(L, POINT)
        per_camera.append([[frame, *_project(L, p0), u1, v1] for frame in range(3)])
    rec, nviews = _run(tmp_path, *_write_cameras(tmp_path, per_camera))

    assert np.allclose(rec[["p0_x", "p0_y", "p0_z"]].to_numpy(), np.tile(p0, (3, 1)))
    assert np.allclose(rec[["p1_x", "p1_y", "p1_z"]].to_numpy(), np.tile(POINT, (3, 1)))
    assert nviews["p0_nviews"].tolist() == [3, 3, 3]
    assert nviews["p1_nviews"].tolist() == [2, 2, 2]


def test_run_reconstruction_marker_seen_by_one_camera_stays_nan(tmp_path):
    p0 = np.array([1.0, 2.0, 3.0])
    per_camera = []
    for i, L in enumerate([CAM_XY, CAM_YZ, CAM_XZ]):
        u1, v1 = _project(L, POINT) if i == 1 else (np.nan, np.nan)  # only camera 1 sees p1
        per_camera.append([[0, *_project(L, p0), u1, v1]])
    rec, nviews = _run(tmp_path, *_write_cameras(tmp_path, per_camera))

    assert rec[["p1_x", "p1_y", "p1_z"]].isna().all().all()
    assert nviews["p1_nviews"].tolist() == [0]
    assert nviews["p0_nviews"].tolist() == [3]
