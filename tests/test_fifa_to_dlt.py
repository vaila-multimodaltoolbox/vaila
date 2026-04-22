"""Tests for vaila.fifa_to_dlt — FIFA cameras/*.npz -> per-frame DLT2D/DLT3D."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vaila.fifa_to_dlt import (
    compute_dlt2d_from_KRt,
    compute_dlt3d_from_KRt,
    convert_cameras_npz_to_dlt,
    undistort_pixel_csv,
)
from vaila.rec2d import rec2d as rec2d_reconstruct
from vaila.rec3d import rec3d_multicam

cv2 = pytest.importorskip("cv2")


def _project_world_to_pixel(
    K: np.ndarray, R: np.ndarray, t: np.ndarray, X: np.ndarray
) -> np.ndarray:
    """X: (N,3) world; returns (N,2) distorted-free pinhole pixels."""
    Xc = (R @ X.T).T + t.reshape(1, 3)
    z = Xc[:, 2:3]
    z = np.where(np.abs(z) < 1e-9, 1e-9, z)
    xn = Xc[:, 0:1] / z
    yn = Xc[:, 1:2] / z
    u = K[0, 0] * xn + K[0, 2]
    v = K[1, 1] * yn + K[1, 2]
    return np.hstack([u, v])


def test_compute_dlt2d_round_trip_pitch_plane() -> None:
    rng = np.random.default_rng(42)
    K = np.array([[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]])
    # Small random rotation + translation (world Z=0 plane)
    angle = 0.15
    c, s = np.cos(angle), np.sin(angle)
    R = np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    ) @ rng.standard_normal((3, 3))
    R, _ = np.linalg.qr(R)
    t = np.array([-2.0, 25.0, 120.0], dtype=np.float64)

    dlt = compute_dlt2d_from_KRt(K, R, t)
    world_xy = np.array([[10.0, 15.0], [40.0, -5.0], [0.0, 0.0], [52.3, 30.1]], dtype=np.float64)
    Zw = np.zeros((len(world_xy), 1))
    Xw = np.hstack([world_xy, Zw])
    pixels = _project_world_to_pixel(K, R, t, Xw)

    for (xw, yw), (px, py) in zip(world_xy, pixels, strict=True):
        out = rec2d_reconstruct(dlt, np.array([[px, py]], dtype=np.float64))
        assert np.allclose(out[0], [xw, yw], atol=1e-4, rtol=1e-5)


def test_compute_dlt3d_round_trip_two_cams() -> None:
    rng = np.random.default_rng(7)
    K1 = np.array([[900.0, 0.0, 512.0], [0.0, 900.0, 384.0], [0.0, 0.0, 1.0]])
    K2 = np.array([[700.0, 0.0, 400.0], [0.0, 700.0, 300.0], [0.0, 0.0, 1.0]])
    R1 = np.eye(3)
    t1 = np.array([0.0, 0.0, 10.0])
    R2, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    t2 = np.array([1.5, -0.5, 8.0])

    pts3d = np.array([[1.0, 2.0, 0.5], [-0.5, 3.0, 1.2], [2.0, -1.0, 2.5]], dtype=np.float64)
    uv1 = _project_world_to_pixel(K1, R1, t1, pts3d)
    uv2 = _project_world_to_pixel(K2, R2, t2, pts3d)

    d1 = compute_dlt3d_from_KRt(K1, R1, t1)
    d2 = compute_dlt3d_from_KRt(K2, R2, t2)

    for i in range(len(pts3d)):
        p3 = rec3d_multicam(
            [d1, d2], [(float(uv1[i, 0]), float(uv1[i, 1])), (float(uv2[i, 0]), float(uv2[i, 1]))]
        )
        assert np.allclose(p3, pts3d[i], atol=1e-3, rtol=1e-4)


def test_convert_cameras_npz_io(tmp_path: Path) -> None:
    n = 5
    K0 = np.diag([500.0, 500.0, 1.0]).copy()
    K0[0, 2] = 320.0
    K0[1, 2] = 240.0
    K = np.stack([K0.copy() for _ in range(n)], axis=0)
    R = np.tile(np.eye(3), (n, 1, 1))
    t = np.tile(np.array([[0.0, 0.0, 5.0]]), (n, 1))
    k = np.zeros((n, 2))
    p = tmp_path / "seq.npz"
    np.savez(
        p,
        K=K.astype(np.float32),
        R=R.astype(np.float32),
        t=t.astype(np.float32),
        k=k.astype(np.float32),
    )

    out = tmp_path / "dlt"
    paths = convert_cameras_npz_to_dlt(p, out, mode="both")
    assert paths["dlt2d"] is not None and paths["dlt3d"] is not None
    df2 = pd.read_csv(paths["dlt2d"])
    df3 = pd.read_csv(paths["dlt3d"])
    assert len(df2) == n and len(df3) == n
    assert list(df2.columns) == ["frame"] + [f"p{i}" for i in range(1, 9)]
    assert list(df3.columns) == ["frame"] + [f"p{i}" for i in range(1, 12)]
    assert (df2["frame"] == np.arange(n)).all()


def test_undistort_pixel_csv_identity_k_zero(tmp_path: Path) -> None:
    n = 3
    K = np.tile(np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]), (n, 1, 1))
    R = np.tile(np.eye(3), (n, 1, 1))
    t = np.tile(np.array([[0.0, 0.0, 10.0]]), (n, 1))
    k = np.zeros((n, 5))
    npz = tmp_path / "cam.npz"
    np.savez(
        npz,
        K=K.astype(np.float32),
        R=R.astype(np.float32),
        t=t.astype(np.float32),
        k=k.astype(np.float32),
    )

    rows = []
    for fi in range(n):
        rows.append({"frame": fi, "p1_x": 100.0 + fi, "p1_y": 200.0 - fi})
    csv_in = tmp_path / "cam.csv"
    pd.DataFrame(rows).to_csv(csv_in, index=False)
    csv_out = tmp_path / "cam_ud.csv"
    undistort_pixel_csv(csv_in, npz, csv_out)
    df0 = pd.read_csv(csv_in)
    df1 = pd.read_csv(csv_out)
    assert np.allclose(
        df0[["p1_x", "p1_y"]].to_numpy(), df1[["p1_x", "p1_y"]].to_numpy(), atol=1e-2
    )
