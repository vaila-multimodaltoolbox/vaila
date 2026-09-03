"""Tests for drawsportsfields robust sport detection and FIFA dataset CSV plotting."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd
import pytest

from vaila import drawsportsfields as dsf

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "vaila" / "models"


def test_detect_sport_all_fifa_and_soccer_variants() -> None:
    """Ensure filenames and contents resolve to their exact sport registry keys."""
    expectations = {
        "soccerfield_ref3d.csv": "soccer",
        "soccerfield_ref3d_fifa.csv": "fifa",
        "soccerfield_ref3d_fifa_center.csv": "fifa_center",
        "soccerfield_ref3d_fifa_dataset.csv": "fifa_dataset",
        "tenniscourt_ref3d.csv": "tennis",
    }
    for filename, expected_sport in expectations.items():
        csv_path = MODELS_DIR / filename
        df = pd.read_csv(csv_path)
        detected = dsf._detect_sport(str(csv_path), df)
        assert (
            detected == expected_sport
        ), f"Expected {expected_sport} for {filename}, got {detected}"


def test_plot_field_with_dataset_csv_no_keyerror() -> None:
    """plot_field must not crash with KeyError: 'midfield_left' when given dataset CSV."""
    csv_path = MODELS_DIR / "soccerfield_ref3d_fifa_dataset.csv"
    df = pd.read_csv(csv_path)
    fig, ax = dsf.plot_field(df)
    assert fig is not None
    assert ax is not None
    matplotlib.pyplot.close(fig)


def test_plot_field_fifa_dataset_with_dataset_csv() -> None:
    """plot_field_fifa_dataset must draw pitch and overlay keypoints without error."""
    csv_path = MODELS_DIR / "soccerfield_ref3d_fifa_dataset.csv"
    df = pd.read_csv(csv_path)
    fig, ax = dsf.plot_field_fifa_dataset(df)
    assert fig is not None
    assert ax is not None
    matplotlib.pyplot.close(fig)


def test_fifa32_dataset_xy_from_field_points_both_formats() -> None:
    """_fifa32_dataset_xy_from_field_points must work on both landmark and dataset files."""
    # 1. 37-point pitch line file (soccerfield_ref3d_fifa.csv)
    df_fifa = pd.read_csv(MODELS_DIR / "soccerfield_ref3d_fifa.csv")
    pts_fifa = {
        str(row["point_name"]): (float(row["x"]), float(row["y"]), int(row["point_number"]))
        for _, row in df_fifa.iterrows()
    }
    xy_derived = dsf._fifa32_dataset_xy_from_field_points(pts_fifa)
    assert xy_derived is not None
    assert len(xy_derived) == 32

    # 2. 48-point dataset file (soccerfield_ref3d_fifa_dataset.csv)
    df_dataset = pd.read_csv(MODELS_DIR / "soccerfield_ref3d_fifa_dataset.csv")
    pts_dataset = {
        str(row["point_name"]): (float(row["x"]), float(row["y"]), int(row["point_number"]))
        for _, row in df_dataset.iterrows()
    }
    xy_direct = dsf._fifa32_dataset_xy_from_field_points(pts_dataset)
    assert xy_direct is not None
    assert len(xy_direct) == 32

    # Both must place keypoints in the exact same coordinates!
    for i in range(32):
        assert pytest.approx(xy_derived[i][0], abs=1e-3) == xy_direct[i][0]
        assert pytest.approx(xy_derived[i][1], abs=1e-3) == xy_direct[i][1]


def test_list_fifa32_control_points_from_dataset_csv() -> None:
    """list_fifa32_control_points must succeed on points from soccerfield_ref3d_fifa_dataset.csv."""
    df = pd.read_csv(MODELS_DIR / "soccerfield_ref3d_fifa_dataset.csv")
    points = {
        str(row["point_name"]): (float(row["x"]), float(row["y"]), int(row["point_number"]))
        for _, row in df.iterrows()
    }
    ctrl_pts = dsf.list_fifa32_control_points(points)
    assert len(ctrl_pts) == 32
    assert ctrl_pts[0].source_name == "top_left_corner"
    assert ctrl_pts[8].source_name == "left_penalty_spot"
