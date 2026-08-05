"""C3D export must flag occluded samples instead of parking them on the origin.

C3D marks an untracked/occluded sample with a NEGATIVE residual (-1). The
exporter used to substitute 0.0 for NaN while leaving the residual valid,
so every occluded marker rendered as a real marker sitting at the world
origin — on real markerless input (SAM3+Sapiens2, 308 keypoints) that was
~16% of all samples collapsed into one spot, which is what a viewer shows
as a dense blob at (0, 0, 0) with lines shooting into it.
"""

import numpy as np
import pandas as pd
import pytest

ezc3d = pytest.importorskip("ezc3d")

try:
    from vaila.readcsv_export import auto_create_c3d_from_csv
except ImportError:  # standalone execution
    from readcsv_export import auto_create_c3d_from_csv  # ty: ignore[unresolved-import]


def _points_df(n_frames=6, n_markers=3):
    data = {"frame": np.arange(n_frames, dtype=float)}
    for m in range(1, n_markers + 1):
        for axis, base in zip(("X", "Y", "Z"), (1.0, 2.0, 3.0), strict=True):
            data[f"p{m}_{axis}"] = base * m + np.arange(n_frames, dtype=float)
    return pd.DataFrame(data)


def _write(df, tmp_path, name="take.c3d", **kwargs):
    out = tmp_path / name
    auto_create_c3d_from_csv(df, str(out), point_rate=100.0, **kwargs)
    return ezc3d.c3d(str(out))


def test_occluded_samples_are_flagged_invalid(tmp_path):
    df = _points_df()
    df.loc[2:3, ["p2_X", "p2_Y", "p2_Z"]] = np.nan

    c3d = _write(df, tmp_path)
    residuals = c3d["data"]["meta_points"]["residuals"]

    assert residuals.shape == (1, 3, 6)
    invalid = residuals[0] < 0
    assert invalid.sum() == 2, "exactly the two occluded samples must be flagged"
    assert invalid[1, 2] and invalid[1, 3]
    # Every other sample stays valid.
    assert not invalid[0].any() and not invalid[2].any()


def test_no_valid_sample_sits_on_the_world_origin(tmp_path):
    df = _points_df()
    df.loc[2:3, ["p2_X", "p2_Y", "p2_Z"]] = np.nan

    c3d = _write(df, tmp_path)
    points = c3d["data"]["points"]
    valid = c3d["data"]["meta_points"]["residuals"][0] >= 0

    at_origin = (np.abs(points[:3]) < 1e-9).all(axis=0)
    assert not at_origin[valid].any(), "occluded marker leaked in as a valid origin point"
    # ezc3d surfaces flagged samples as NaN on read-back, which is what a
    # viewer needs in order to skip them.
    assert np.isnan(points[:3, 1, 2]).all()


def test_fully_tracked_trial_flags_nothing(tmp_path):
    c3d = _write(_points_df(), tmp_path)
    assert (c3d["data"]["meta_points"]["residuals"] >= 0).all()
    assert np.isfinite(c3d["data"]["points"][:3]).all()


def test_valid_coordinates_survive_unchanged(tmp_path):
    df = _points_df()
    df.loc[2, ["p1_X", "p1_Y", "p1_Z"]] = np.nan

    c3d = _write(df, tmp_path)
    points = c3d["data"]["points"]

    # Marker p3 is untouched by the occlusion of p1.
    np.testing.assert_allclose(points[0, 2, :], df["p3_X"].to_numpy(), atol=1e-5)
    np.testing.assert_allclose(points[1, 2, :], df["p3_Y"].to_numpy(), atol=1e-5)
    np.testing.assert_allclose(points[2, 2, :], df["p3_Z"].to_numpy(), atol=1e-5)


def test_unit_conversion_still_applies_with_occlusion(tmp_path):
    df = _points_df()
    df.loc[1, ["p1_X", "p1_Y", "p1_Z"]] = np.nan

    c3d = _write(df, tmp_path, name="take_mm.c3d", conversion_factor=1000, point_units="mm")

    assert c3d["parameters"]["POINT"]["UNITS"]["value"] == ["mm"]
    assert (c3d["data"]["meta_points"]["residuals"] < 0).sum() == 1
    np.testing.assert_allclose(
        c3d["data"]["points"][0, 2, :], df["p3_X"].to_numpy() * 1000, atol=1e-2
    )
