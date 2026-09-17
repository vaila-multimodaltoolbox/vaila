"""Tests for drawsportsfields REF3D export (DLT3D control-point selection)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from vaila import drawsportsfields as dsf
from vaila.dlt3d import read_ref3d_file


def _fifa_points() -> dict[str, tuple[float, float, int]]:
    csv_path = (
        Path(__file__).resolve().parents[1] / "vaila" / "models" / "soccerfield_ref3d_fifa.csv"
    )
    df = pd.read_csv(csv_path)
    return {
        str(row["point_name"]): (float(row["x"]), float(row["y"]), int(row["point_number"]))
        for _, row in df.iterrows()
    }


def test_kiki_export_defaults_to_49_zero_based_points(tmp_path: Path) -> None:
    csv_path = (
        Path(__file__).resolve().parents[1] / "vaila" / "models" / "soccerfield_kiki.csv"
    )
    pts = dsf.list_model_control_points(pd.read_csv(csv_path))

    assert len(pts) == 49
    assert [pt.source_index for pt in pts] == list(range(49))

    written = dsf.write_ref3d_export(tmp_path / "kiki_selected.ref3d", pts)
    ref_df = pd.read_csv(written["ref3d"])
    map_df = pd.read_csv(written["map"])
    pixel_df = pd.read_csv(written["pixel_template"])

    assert list(ref_df.columns)[1:4] == ["p0_x", "p0_y", "p0_z"]
    assert list(ref_df.columns)[-3:] == ["p48_x", "p48_y", "p48_z"]
    assert len(ref_df.columns) == 1 + 49 * 3
    assert map_df["p_index"].tolist() == list(range(49))
    assert list(pixel_df.columns)[1:3] == ["p0_x", "p0_y"]
    assert list(pixel_df.columns)[-2:] == ["p48_x", "p48_y"]
    assert len(pixel_df.columns) == 1 + 49 * 2


def test_legacy_one_based_export_remains_readable_by_dlt3d(tmp_path: Path) -> None:
    pts = dsf.list_fifa32_control_points(_fifa_points())
    assert len(pts) == 32
    # Corners + midfield + spots (8 points, non-coplanar only if Z varies; Z=0 here
    # but file format must still be valid for dlt3d.read_ref3d_file).
    chosen = [pts[i] for i in (0, 5, 8, 13, 16, 21, 24, 29)]
    written = dsf.write_ref3d_export(tmp_path / "pitch_sel.ref3d", chosen, index_base=1)
    assert written["ref3d"].is_file()
    assert written["map"].is_file()
    assert written["pixel_template"].is_file()

    ref_df = read_ref3d_file(str(written["ref3d"]))
    assert ref_df is not None
    assert list(ref_df.columns)[:1] == ["frame"]
    assert "p1_x" in ref_df.columns and "p8_z" in ref_df.columns
    assert float(ref_df.iloc[0]["p1_x"]) == pts[0].x
    assert float(ref_df.iloc[0]["p3_x"]) == pts[8].x  # left_penalty_spot

    map_df = pd.read_csv(written["map"])
    assert list(map_df["source_name"])[2] == "left_penalty_spot"
    assert int(map_df.iloc[2]["p_index"]) == 3


def test_list_model_control_points_from_fifa_csv() -> None:
    csv_path = (
        Path(__file__).resolve().parents[1] / "vaila" / "models" / "soccerfield_ref3d_fifa.csv"
    )
    df = pd.read_csv(csv_path)
    pts = dsf.list_model_control_points(df)
    assert len(pts) == 37
    assert pts[0].source_name == "bottom_left_corner"
    assert pts[0].source_index == 0
    assert pts[21].source_name == "left_penalty_spot"  # point_number 21 (0-based)


def test_model_without_point_numbers_defaults_to_zero_based_indices() -> None:
    model = pd.DataFrame(
        {
            "point_name": ["first", "second"],
            "x": [0.0, 1.0],
            "y": [0.0, 1.0],
            "z": [0.0, 0.0],
        }
    )
    pts = dsf.list_model_control_points(model)
    assert [point.source_index for point in pts] == [0, 1]
    assert [point.label for point in pts] == ["00  first", "01  second"]


def test_write_ref3d_rejects_too_few_points(tmp_path: Path) -> None:
    pts = dsf.list_fifa32_control_points(_fifa_points())[:3]
    try:
        dsf.write_ref3d_export(tmp_path / "too_few.ref3d", pts)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "at least 6" in str(exc).lower()
