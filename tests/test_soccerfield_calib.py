"""Smoke tests for vaila.soccerfield_calib (DLT2D from the FIFA reference)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vaila import soccerfield_calib as sc

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REF = REPO_ROOT / "models" / "soccerfield_ref3d.csv"


def test_load_field_reference_has_29_points() -> None:
    kps = sc.load_field_reference(DEFAULT_REF)
    assert len(kps) >= 29
    names = {kp.name for kp in kps}
    for must_have in (
        "bottom_left_corner",
        "bottom_right_corner",
        "top_left_corner",
        "top_right_corner",
        "center_field",
        "left_penalty_spot",
        "right_penalty_spot",
    ):
        assert must_have in names


def test_compute_dlt2d_recovers_identity() -> None:
    # When world and pixel coordinates are identical (scaled), the reprojection
    # error must be essentially zero — a sanity check for the DLT math path.
    world = np.array(
        [
            [0.0, 0.0],
            [105.0, 0.0],
            [105.0, 68.0],
            [0.0, 68.0],
            [52.5, 34.0],
            [52.5, 0.0],
            [52.5, 68.0],
        ],
        dtype=float,
    )
    pixels = world * 10.0 + np.array([20.0, 30.0])  # simple affine
    params, rms, per_point = sc.compute_dlt2d(pixels, world)
    assert params.shape == (8,)
    assert per_point.shape == (7,)
    assert rms < 1e-6, f"expected near-zero RMS error, got {rms}"


def test_compute_dlt2d_raises_on_few_points() -> None:
    world = np.zeros((5, 2))
    pixels = np.zeros((5, 2))
    with pytest.raises(ValueError):
        sc.compute_dlt2d(pixels, world)


def test_end_to_end_with_pixel_csv(tmp_path: Path) -> None:
    # Build a synthetic pixel CSV using the real FIFA reference to exercise
    # load_pixel_points + run_soccerfield_calib (no video, no GUI).
    kps_all = sc.load_field_reference(DEFAULT_REF)
    selected = [
        kp
        for kp in kps_all
        if kp.name
        in {
            "bottom_left_corner",
            "bottom_right_corner",
            "top_left_corner",
            "top_right_corner",
            "center_field",
            "left_penalty_spot",
            "right_penalty_spot",
        }
    ]
    world = np.array([kp.world_xy for kp in selected])
    # pretend the camera is a perfect scale+shift ("ortho-broadcast")
    pixels = world * np.array([10.0, 12.0]) + np.array([40.0, 80.0])

    import pandas as pd

    row = {"frame": 0}
    names = [kp.name for kp in selected]
    for name, (x, y) in zip(names, pixels, strict=False):
        row[f"{name}_x"] = x
        row[f"{name}_y"] = y
    csv_path = tmp_path / "pixels.csv"
    pd.DataFrame([row]).to_csv(csv_path, index=False)

    out_dir = tmp_path / "out"
    data_root = tmp_path / "data"
    res = sc.run_soccerfield_calib(
        video=None,
        ref3d_csv=DEFAULT_REF,
        pixel_csv=csv_path,
        output_dir=out_dir,
        data_root=data_root,
    )

    assert Path(res["dlt2d"]).exists()
    assert Path(res["ref2d"]).exists()
    assert Path(res["homography_report"]).exists()
    assert Path(res["cameras_npz"]).exists()
    assert res["n_points"] == len(selected)
    assert res["rms_pixels_or_metres"] < 1e-6
