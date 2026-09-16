"""Synthetic CPU-only tests for vaila.monocular_planar_align."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

try:
    from vaila.monocular_planar_align import (
        align_monocular_to_planar_world,
        pixels_to_floor_xy,
        smooth_scales,
        solve_planar_similarity,
    )
    from vaila.planar_geometry_tracker import project_points
except ImportError:
    from monocular_planar_align import (  # ty: ignore[unresolved-import]
        align_monocular_to_planar_world,
        pixels_to_floor_xy,
        smooth_scales,
        solve_planar_similarity,
    )
    from planar_geometry_tracker import project_points  # ty: ignore[unresolved-import]


def test_pixels_to_floor_xy_identity_homography():
    h_inv = np.eye(3)
    uv = np.array([[1.5, 2.5], [0.0, 0.0]], dtype=np.float64)
    xy = pixels_to_floor_xy(h_inv, uv)
    np.testing.assert_allclose(xy, uv, atol=1e-12)


def test_solve_planar_similarity_recovers_known_similarity():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(8, 3))
    # Non-planar source cloud.
    src[:, 2] += np.linspace(-0.2, 0.2, 8)
    r_true = Rotation.from_euler("xyz", [10, -5, 30], degrees=True).as_matrix()
    s_true = 1.37
    t_true = np.array([0.4, -0.2, 0.05])
    tgt = (s_true * (src @ r_true.T)) + t_true

    out = solve_planar_similarity(src, tgt, min_points=4)
    assert out is not None
    r_fit, s_fit, t_fit = out
    assert s_fit == pytest.approx(s_true, rel=1e-6)
    np.testing.assert_allclose(r_fit, r_true, atol=1e-6)
    np.testing.assert_allclose(t_fit, t_true, atol=1e-6)


def test_smooth_scales_keeps_positive_and_reduces_jitter():
    scales = np.array([1.0, 1.4, 0.9, 1.3, 1.0, 1.2, 0.95, 1.15, 1.05, 1.1] * 5)
    smoothed = smooth_scales(scales, cutoff_hz=3.0, fps=60.0)
    assert np.all(smoothed > 0)
    assert float(np.std(smoothed)) < float(np.std(scales))


def test_align_monocular_to_planar_world_end_to_end(tmp_path: Path):
    """Synthetic body + identity H recovers feet near Z=0 and known XY."""
    try:
        from vaila.monocular_planar_align import _rotation_camera_up_to_world_z
    except ImportError:
        from monocular_planar_align import (  # ty: ignore[unresolved-import]
            _rotation_camera_up_to_world_z,
        )

    n_frames = 20
    n_markers = 70
    body = np.zeros((n_markers, 3), dtype=np.float64)
    body[0] = [0.0, -0.9, 2.0]
    body[9] = [-0.10, -0.10, 2.0]
    body[10] = [0.10, -0.10, 2.0]
    body[13] = [-0.12, 0.85, 1.95]
    body[14] = [0.12, 0.85, 1.95]
    body[17] = [-0.12, 0.90, 1.90]
    body[20] = [0.12, 0.90, 1.90]

    r_up = _rotation_camera_up_to_world_z()
    s_true = 0.85
    t_true = np.array([0.30, 0.40, 0.0])
    origin = 0.5 * (body[9] + body[10])
    shape = body - origin
    world_body = s_true * (shape @ r_up.T) + t_true
    for mi in (13, 14, 17, 20):
        world_body[mi, 2] = 0.0
        # Rebuild targets as the exact Umeyama destination for camera feet:
        # use planted XY but keep correspondence consistent with a fresh solve.
    # Recompute targets from planted XY (Z=0) — pixels = those XY under identity H.
    planted = {
        13: np.array([0.18, 0.22, 0.0]),
        14: np.array([0.42, 0.22, 0.0]),
        17: np.array([0.16, 0.05, 0.0]),
        20: np.array([0.44, 0.05, 0.0]),
    }
    # Fit known similarity from camera foot shapes onto planted targets, apply to body.
    src_feet = np.stack([shape[mi] for mi in planted])
    tgt_feet = np.stack([planted[mi] for mi in planted])
    solved = solve_planar_similarity(src_feet, tgt_feet, min_points=4)
    assert solved is not None
    r_true, s_true, t_true = solved
    world_body = s_true * (shape @ r_true.T) + t_true

    h = np.eye(3)
    h_inv = np.eye(3)
    frames = np.arange(n_frames, dtype=np.int64)
    cam_stack = np.tile(body[None, :, :], (n_frames, 1, 1))
    uv_stack = np.full((n_frames, n_markers, 2), np.nan)
    for mi, xyz in planted.items():
        uv_stack[:, mi, :] = xyz[:2]

    mono_path = tmp_path / "toy_mhr70_rec3d.csv"
    pix_path = tmp_path / "toy_markers.csv"
    header3 = ["frame"] + [f"p{m}_{a}" for m in range(1, n_markers + 1) for a in "xyz"]
    header2 = ["frame"] + [f"p{m}_{a}" for m in range(1, n_markers + 1) for a in "xy"]
    rows3 = np.column_stack([frames.astype(float), cam_stack.reshape(n_frames, -1)])
    rows2 = np.column_stack([frames.astype(float), uv_stack.reshape(n_frames, -1)])
    pd.DataFrame(rows3, columns=header3).to_csv(mono_path, index=False)
    pd.DataFrame(rows2, columns=header2).to_csv(pix_path, index=False)

    npz_path = tmp_path / "homographies.npz"
    np.savez_compressed(
        npz_path,
        H=np.stack([h] * n_frames),
        H_inv=np.stack([h_inv] * n_frames),
        frame_ids=frames,
    )

    result = align_monocular_to_planar_world(
        mono_path,
        npz_path,
        tmp_path / "out",
        pixels_path=pix_path,
        point_rate=60.0,
        smooth_hz=0.0,
        export_mesh="none",
        expected_height_m=None,
        gui=False,
    )
    assert result is not None
    out_dir, file_base = result
    df = pd.read_csv(Path(out_dir) / f"{file_base}.csv")
    # 1-based p14 = 0-based marker 13 (left ankle)
    assert float(np.nanmean(np.abs(df["p14_z"].to_numpy()))) < 0.08
    np.testing.assert_allclose(df["p14_x"].mean(), planted[13][0], atol=0.08)
    np.testing.assert_allclose(df["p14_y"].mean(), planted[13][1], atol=0.08)
    assert (Path(out_dir) / "README_monocular_planar_align.txt").is_file()
    # Recovered scale should be near the planted similarity.
    align_csv = pd.read_csv(Path(out_dir) / f"{file_base}_alignment.csv")
    assert float(align_csv["scale"].mean()) == pytest.approx(s_true, rel=0.05)


def test_project_points_roundtrip_with_npz_style_h():
    """H world→pixel and H_inv pixel→world round-trip."""
    # Simple scale+translate H.
    h = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 20.0], [0.0, 0.0, 1.0]])
    h_inv = np.linalg.inv(h)
    world = np.array([[0.0, 0.0], [0.5, 0.25], [1.0, 1.0]])
    pix = project_points(h, world)
    back = pixels_to_floor_xy(h_inv, pix)
    np.testing.assert_allclose(back, world, atol=1e-10)
