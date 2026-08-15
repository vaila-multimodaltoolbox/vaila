"""Synthetic tests for field-plane + Vitruvian vertical DLT3D calibration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vaila.soccerfield_vitruvian_dlt3d import (
    assign_track_heights,
    calibrate_time_varying,
    fit_vitruvian_frame,
    load_wide_points,
    project_dlt3d,
    solve_vertical_dlt_column,
)

TRUE_DLT = np.array([12.0, 2.0, 4.0, 640.0, 1.0, -8.0, -6.0, 360.0, 0.01, 0.02, 0.04])
FIELD_XY = np.array(
    [
        [-10.0, -8.0],
        [-10.0, 0.0],
        [-10.0, 8.0],
        [0.0, -8.0],
        [0.0, 0.0],
        [0.0, 8.0],
        [10.0, -8.0],
        [10.0, 8.0],
    ]
)
BASE_XY = np.array([[-6.0, -3.0], [2.0, 5.0], [7.0, -5.0]])
HEIGHTS = np.array([1.70, 1.80, 1.90])


def _project_plane(xy: np.ndarray) -> np.ndarray:
    return project_dlt3d(TRUE_DLT, np.column_stack((xy, np.zeros(len(xy)))))


def test_fit_vitruvian_frame_recovers_full_dlt3d() -> None:
    field_pixels = _project_plane(FIELD_XY)
    bottom = _project_plane(BASE_XY)
    top = project_dlt3d(TRUE_DLT, np.column_stack((BASE_XY, HEIGHTS)))

    recovered, diagnostics, controls = fit_vitruvian_frame(
        frame=12,
        field_world_xy=FIELD_XY,
        field_pixels=field_pixels,
        bbox_bottom_pixels=bottom,
        bbox_top_pixels=top,
        bbox_heights_m=HEIGHTS,
        bbox_names=["p1", "p2", "p3"],
    )

    assert np.allclose(recovered, TRUE_DLT, atol=1.0e-8)
    assert diagnostics.vertical_rank == 3
    assert diagnostics.field_reprojection_rms_px < 1.0e-8
    assert diagnostics.vertical_reprojection_rms_px < 1.0e-8
    assert len(controls) == 3
    assert all(row["source"] == "bbox_vitruvian" for row in controls)


def test_one_vertical_does_not_close_dlt3d() -> None:
    planar = np.array(
        [
            TRUE_DLT[0],
            TRUE_DLT[1],
            TRUE_DLT[3],
            TRUE_DLT[4],
            TRUE_DLT[5],
            TRUE_DLT[7],
            TRUE_DLT[8],
            TRUE_DLT[9],
        ]
    )
    base = BASE_XY[:1]
    top = project_dlt3d(TRUE_DLT, np.column_stack((base, HEIGHTS[:1])))
    with pytest.raises(ValueError, match="at least 2"):
        solve_vertical_dlt_column(planar, base, top, HEIGHTS[:1])


def test_height_assignment_supports_explicit_and_ranked_csv(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "frame": [0, 1],
            "x1": [100.0, 100.0],
            "y1": [300.0, 310.0],
            "x2": [200.0, 200.0],
            "y2": [280.0, 290.0],
        }
    ).to_csv(tmp_path / "bottom.csv", index=False)
    pd.DataFrame(
        {
            "frame": [0, 1],
            "x1": [100.0, 100.0],
            "y1": [100.0, 110.0],
            "x2": [200.0, 200.0],
            "y2": [140.0, 150.0],
        }
    ).to_csv(tmp_path / "top.csv", index=False)
    bottom = load_wide_points(tmp_path / "bottom.csv")
    top = load_wide_points(tmp_path / "top.csv")

    pd.DataFrame({"track": ["p1", "p2"], "height_m": [1.91, 1.73]}).to_csv(
        tmp_path / "explicit.csv", index=False
    )
    explicit = assign_track_heights(
        bottom, top, heights_csv=tmp_path / "explicit.csv", default_height_m=None
    )
    assert explicit["p1"].height_m == pytest.approx(1.91)
    assert explicit["p1"].method == "explicit_track"

    pd.DataFrame({"nome": ["Tall", "Short"], "altura_m": [1.95, 1.70]}).to_csv(
        tmp_path / "roster.csv", index=False
    )
    ranked = assign_track_heights(
        bottom, top, heights_csv=tmp_path / "roster.csv", default_height_m=None
    )
    assert ranked["p1"].height_m == pytest.approx(1.95)
    assert ranked["p2"].height_m == pytest.approx(1.70)
    assert ranked["p1"].method == "bbox_height_rank"


def test_time_varying_csv_writes_one_dlt_per_supported_frame(tmp_path: Path) -> None:
    field_ref = pd.DataFrame(
        {
            "point_name": [f"field_{idx}" for idx in range(len(FIELD_XY))],
            "x": FIELD_XY[:, 0],
            "y": FIELD_XY[:, 1],
            "z": 0.0,
        }
    )
    field_ref.to_csv(tmp_path / "field_ref.csv", index=False)

    field_rows: list[dict[str, float]] = []
    bottom_rows: list[dict[str, float]] = []
    top_rows: list[dict[str, float]] = []
    for frame in (0, 1):
        camera = TRUE_DLT.copy()
        camera[3] += frame * 5.0
        field_pixels = project_dlt3d(camera, np.column_stack((FIELD_XY, np.zeros(len(FIELD_XY)))))
        bottoms = project_dlt3d(camera, np.column_stack((BASE_XY, np.zeros(len(BASE_XY)))))
        tops = project_dlt3d(camera, np.column_stack((BASE_XY, HEIGHTS)))
        field_row: dict[str, float] = {"frame": float(frame)}
        bottom_row: dict[str, float] = {"frame": float(frame)}
        top_row: dict[str, float] = {"frame": float(frame)}
        for idx, point in enumerate(field_pixels, start=1):
            field_row[f"p{idx}_x"], field_row[f"p{idx}_y"] = point
        for idx, (bottom, top) in enumerate(zip(bottoms, tops, strict=True), start=1):
            bottom_row[f"x{idx}"], bottom_row[f"y{idx}"] = bottom
            top_row[f"x{idx}"], top_row[f"y{idx}"] = top
        field_rows.append(field_row)
        bottom_rows.append(bottom_row)
        top_rows.append(top_row)
    pd.DataFrame(field_rows).to_csv(tmp_path / "field.csv", index=False)
    pd.DataFrame(bottom_rows).to_csv(tmp_path / "bottom.csv", index=False)
    pd.DataFrame(top_rows).to_csv(tmp_path / "top.csv", index=False)
    pd.DataFrame({"track": ["p1", "p2", "p3"], "height_m": HEIGHTS}).to_csv(
        tmp_path / "heights.csv", index=False
    )

    dlt_df, report_df, controls_df, heights_df = calibrate_time_varying(
        field_pixels_csv=tmp_path / "field.csv",
        field_reference_csv=tmp_path / "field_ref.csv",
        bbox_bottom_csv=tmp_path / "bottom.csv",
        bbox_top_csv=tmp_path / "top.csv",
        heights_csv=tmp_path / "heights.csv",
        default_height_m=None,
    )

    assert dlt_df["frame"].tolist() == [0, 1]
    assert np.allclose(dlt_df.iloc[0, 1:].to_numpy(dtype=float), TRUE_DLT, atol=1.0e-8)
    assert report_df["status"].tolist() == ["ok", "ok"]
    assert len(controls_df) == 6
    assert set(heights_df["method"]) == {"explicit_track"}
