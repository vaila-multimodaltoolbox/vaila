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


def test_all_formats_normalize_to_identical_format1() -> None:
    df1 = normalize_ref3d_to_format1(str(FORMAT1))
    df2 = normalize_ref3d_to_format1(str(FORMAT2))
    df3 = normalize_ref3d_to_format1(str(FORMAT3))
    assert df1 is not None and df2 is not None and df3 is not None

    pd.testing.assert_frame_equal(df1, df2)
    pd.testing.assert_frame_equal(df1, df3)

    assert df1.shape == (1, 76)  # frame + 25 points × 3 axes
    assert float(df1.iloc[0]["p25_z"]) == pytest.approx(1.19)


def test_read_ref3d_file_returns_format1_for_all_variants() -> None:
    ref1 = read_ref3d_file(str(FORMAT1))
    ref2 = read_ref3d_file(str(FORMAT2))
    ref3 = read_ref3d_file(str(FORMAT3))
    assert ref1 is not None and ref2 is not None and ref3 is not None
    pd.testing.assert_frame_equal(ref1, ref2)
    pd.testing.assert_frame_equal(ref1, ref3)


def test_process_files_yields_identical_dlt_for_all_ref3d_formats() -> None:
    dlt1 = process_files(str(PIXEL_FILE), str(FORMAT1))
    dlt2 = process_files(str(PIXEL_FILE), str(FORMAT2))
    dlt3 = process_files(str(PIXEL_FILE), str(FORMAT3))
    assert dlt1 is not None and dlt2 is not None and dlt3 is not None

    frames = sorted(dlt1.keys())
    assert frames == sorted(dlt2.keys()) == sorted(dlt3.keys())
    for frame in frames:
        np.testing.assert_allclose(dlt1[frame], dlt2[frame], rtol=1e-9, atol=1e-6)
        np.testing.assert_allclose(dlt1[frame], dlt3[frame], rtol=1e-9, atol=1e-6)


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
