"""Tests for dlt3d REF3D multi-format loading (formats 1, 2, 3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vaila.dlt3d import (
    detect_ref3d_format,
    normalize_ref3d_to_format1,
    process_files,
    read_ref3d_file,
)

FIXTURE_DIR = Path(__file__).resolve().parent / "DLT3D_and_Rec3d" / "ref3d_realworld"
FORMAT1 = FIXTURE_DIR / "ref3d_realworld_format1.ref3d"
FORMAT2 = FIXTURE_DIR / "ref3d_realworld_format2.ref3d"
FORMAT3 = FIXTURE_DIR / "ref3d_realworld_format3.ref3d"
PIXEL_FILE = (
    Path(__file__).resolve().parent / "DLT3D_and_Rec3d" / "pixelcorrds" / "c01_markers_1_line.csv"
)


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (FORMAT1, 1),
        (FORMAT2, 2),
        (FORMAT3, 3),
    ],
)
def test_detect_ref3d_format(path: Path, expected: int) -> None:
    assert detect_ref3d_format(str(path)) == expected


def test_explicit_legacy_formats_preserve_their_one_based_indices() -> None:
    df1 = normalize_ref3d_to_format1(str(FORMAT1))
    df3 = normalize_ref3d_to_format1(str(FORMAT3))
    assert df1 is not None and df3 is not None

    pd.testing.assert_frame_equal(df1, df3)
    assert df1.shape == (1, 76)  # frame + 25 points × 3 axes
    assert "p1_x" in df1.columns and "p25_z" in df1.columns
    assert float(df1.iloc[0]["p25_z"]) == pytest.approx(1.19)


def test_implicit_format2_uses_vaila_zero_based_order() -> None:
    df2 = normalize_ref3d_to_format1(str(FORMAT2))
    assert df2 is not None
    assert df2.shape == (1, 76)  # frame + 25 points × 3 axes
    assert "p0_x" in df2.columns and "p24_z" in df2.columns
    assert float(df2.iloc[0]["p24_z"]) == pytest.approx(1.19)


def test_read_ref3d_file_preserves_explicit_and_implicit_bases() -> None:
    ref1 = read_ref3d_file(str(FORMAT1))
    ref2 = read_ref3d_file(str(FORMAT2))
    ref3 = read_ref3d_file(str(FORMAT3))
    assert ref1 is not None and ref2 is not None and ref3 is not None
    pd.testing.assert_frame_equal(ref1, ref3)
    assert "p1_x" in ref1.columns and "p25_z" in ref1.columns
    assert "p0_x" in ref2.columns and "p24_z" in ref2.columns


def test_process_files_preserves_legacy_explicit_formats() -> None:
    dlt1 = process_files(str(PIXEL_FILE), str(FORMAT1))
    dlt3 = process_files(str(PIXEL_FILE), str(FORMAT3))
    assert dlt1 is not None and dlt3 is not None

    frames = sorted(dlt1.keys())
    assert frames == sorted(dlt3.keys())
    for frame in frames:
        np.testing.assert_allclose(dlt1[frame], dlt3[frame], rtol=1e-9, atol=1e-6)


def test_process_files_rejects_one_based_pixels_with_zero_based_ref3d(capsys) -> None:
    assert process_files(str(PIXEL_FILE), str(FORMAT2)) is None
    output = capsys.readouterr().out
    assert "shifted by -1" in output
    assert "vaila standard: p0" in output


def test_process_files_accepts_zero_based_pixels_with_implicit_ref3d(tmp_path: Path) -> None:
    pixel_df = pd.read_csv(PIXEL_FILE)
    renamed = {}
    for column in pixel_df.columns:
        if column.startswith("p") and "_" in column:
            point, axis = column.split("_", maxsplit=1)
            renamed[column] = f"p{int(point[1:]) - 1}_{axis}"
    zero_pixel_file = tmp_path / "pixels_zero_based.csv"
    pixel_df.rename(columns=renamed).to_csv(zero_pixel_file, index=False)

    legacy_dlt = process_files(str(PIXEL_FILE), str(FORMAT1))
    zero_dlt = process_files(str(zero_pixel_file), str(FORMAT2))
    assert legacy_dlt is not None and zero_dlt is not None
    assert sorted(legacy_dlt) == sorted(zero_dlt)
    for frame in legacy_dlt:
        np.testing.assert_allclose(legacy_dlt[frame], zero_dlt[frame], rtol=1e-9, atol=1e-6)


def test_format3_uses_index_column_not_row_order(tmp_path: Path) -> None:
    """Shuffled rows must still map pN via the index column."""
    shuffled = tmp_path / "shuffled.ref3d"
    shuffled.write_text(
        "3,0.0,0.0,0.545\n"
        "1,0.0,0.0,0.0\n"
        "2,0.0,0.0,0.285\n"
        "4,0.0,0.0,0.83\n"
        "5,0.0,0.0,1.185\n"
        "6,0.0,4.877,0.0\n",
        encoding="utf-8",
    )
    df = normalize_ref3d_to_format1(str(shuffled))
    assert df is not None
    assert float(df.iloc[0]["p1_z"]) == pytest.approx(0.0)
    assert float(df.iloc[0]["p3_z"]) == pytest.approx(0.545)


def test_normalize_rejects_too_few_points(tmp_path: Path) -> None:
    tiny = tmp_path / "tiny.ref3d"
    tiny.write_text("0.0,0.0,0.0\n1.0,0.0,0.0\n", encoding="utf-8")
    assert normalize_ref3d_to_format1(str(tiny)) is None


def test_headed_long_format4_point_xyz(tmp_path: Path) -> None:
    """Headed ``point,x,y,z`` preserves explicit zero-based point ids."""
    headed = tmp_path / "mode2.ref3d"
    headed.write_text(
        "point,x,y,z\n"
        "0,0.0,0.0,0.0\n"
        "1,1.0,0.0,0.0\n"
        "2,1.0,1.0,0.0\n"
        "3,0.0,1.0,0.0\n"
        "4,0.5,0.5,1.0\n"
        "5,0.5,0.5,0.5\n",
        encoding="utf-8",
    )
    assert detect_ref3d_format(str(headed)) == 4
    df = normalize_ref3d_to_format1(str(headed), min_points=4)
    assert df is not None
    assert float(df.iloc[0]["p0_x"]) == pytest.approx(0.0)
    assert float(df.iloc[0]["p1_x"]) == pytest.approx(1.0)
    assert "p5_z" in df.columns
