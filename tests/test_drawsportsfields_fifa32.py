"""Tests for FIFA Dataset Labeling (32 KP) overlay placement on pitch lines."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from vaila import drawsportsfields as dsf


def _load_fifa_points() -> dict[str, tuple[float, float, int]]:
    csv_path = (
        Path(__file__).resolve().parents[1] / "vaila" / "models" / "soccerfield_ref3d_fifa.csv"
    )
    df = pd.read_csv(csv_path)
    return {
        str(row["point_name"]): (float(row["x"]), float(row["y"]), int(row["point_number"]))
        for _, row in df.iterrows()
    }


def test_fifa32_overlay_sits_on_pitch_lines() -> None:
    """Interior KPs must match FIFA geometry, not Roboflow-stretched corners."""
    points = _load_fifa_points()
    xy = dsf._fifa32_dataset_xy_from_field_points(points)
    assert xy is not None
    assert len(xy) == 32

    # Border corners
    assert xy[0] == (points["top_left_corner"][0], points["top_left_corner"][1])
    assert xy[5] == (points["bottom_left_corner"][0], points["bottom_left_corner"][1])

    # User-flagged interior points (must sit on drawn lines)
    assert xy[8] == (points["left_penalty_spot"][0], points["left_penalty_spot"][1])
    assert xy[8] == (-41.45, 0.0)
    assert xy[9] == (
        points["left_penalty_area_top_right"][0],
        points["left_penalty_area_top_right"][1],
    )
    assert xy[12] == (
        points["left_penalty_area_top_left"][0],
        points["left_penalty_area_top_left"][1],
    )
    assert xy[14] == (
        points["center_circle_top_intersection"][0],
        points["center_circle_top_intersection"][1],
    )
    assert xy[15] == (
        points["center_circle_bottom_intersection"][0],
        points["center_circle_bottom_intersection"][1],
    )
    assert xy[17] == (
        points["right_penalty_area_top_right"][0],
        points["right_penalty_area_top_right"][1],
    )
    assert xy[30] == (-9.15, 0.0)
    assert xy[31] == (9.15, 0.0)

    # Pen-box inner at goal-area Y
    assert xy[10] == (-35.95, 9.16)
    assert xy[11] == (-35.95, -9.16)
