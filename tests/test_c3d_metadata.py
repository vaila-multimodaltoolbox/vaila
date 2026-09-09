"""Tests for C3D metadata reader, editor, creator, and CLI."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from vaila.c3d_metadata import (
    create_c3d,
    format_c3d_metadata_summary,
    read_c3d_metadata,
    update_c3d_metadata,
)
from vaila.c3d_metadata import (
    main as c3d_metadata_main,
)

FIXTURE_C3D = Path(__file__).parent / "C3D_to_CSV" / "C3D_to_CSV_01.c3d"


def test_read_c3d_metadata_fixture():
    assert FIXTURE_C3D.is_file(), f"Fixture missing: {FIXTURE_C3D}"
    meta = read_c3d_metadata(FIXTURE_C3D)

    assert meta["file_name"] == "C3D_to_CSV_01.c3d"
    assert meta["point_rate"] == 100.0
    assert meta["point_units"] == "mm"
    assert meta["num_markers"] == 166
    assert meta["num_frames"] == 91
    assert meta["analog_rate"] == 2000.0
    assert meta["num_analogs"] == 254
    assert meta["subframe_ratio"] == 20.0
    assert "Vicon" in meta["manufacturer"]["company"]

    summary = format_c3d_metadata_summary(meta)
    assert "Point Frequency:    100.00 Hz" in summary
    assert "Point Units:        mm" in summary
    assert "Number of Markers:  166" in summary


def test_create_c3d(tmp_path: Path):
    out_c3d = tmp_path / "new_template.c3d"
    result = create_c3d(
        out_c3d,
        num_frames=50,
        point_rate=200.0,
        point_units="m",
        num_markers=6,
        num_analogs=8,
        analog_rate=2000.0,
        manufacturer="vailá custom",
        software="vailá v0.3.131",
    )
    assert os.path.isfile(result)

    meta = read_c3d_metadata(out_c3d)
    assert meta["num_frames"] == 50
    assert meta["point_rate"] == 200.0
    assert meta["point_units"] == "m"
    assert meta["num_markers"] == 6
    assert meta["num_analogs"] == 8
    assert meta["analog_rate"] == 2000.0
    assert meta["manufacturer"]["company"] == "vailá custom"
    assert meta["manufacturer"]["software"] == "vailá v0.3.131"


def test_update_c3d_metadata(tmp_path: Path):
    out_c3d = tmp_path / "updated.c3d"
    result = update_c3d_metadata(
        FIXTURE_C3D,
        out_c3d,
        point_rate=120.0,
        point_units="m",
        scale_coordinates=True,
        manufacturer="Qualisys",
        software="QTM 2026",
    )
    assert os.path.isfile(result)

    meta = read_c3d_metadata(out_c3d)
    assert meta["point_rate"] == 120.0
    assert meta["point_units"] == "m"
    assert meta["manufacturer"]["company"] == "Qualisys"
    assert meta["manufacturer"]["software"] == "QTM 2026"
    assert meta["num_markers"] == 166
    assert meta["num_frames"] == 91
    # Analog rate should have stayed proportional to subframe ratio (20x -> 2400 Hz)
    assert meta["analog_rate"] == 2400.0


def test_c3d_metadata_cli_modes(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    # Mode 1: show
    code_show = c3d_metadata_main(["--show", str(FIXTURE_C3D)])
    assert code_show == 0
    captured = capsys.readouterr()
    assert "C3D File: C3D_to_CSV_01.c3d" in captured.out

    # Mode 2: create
    cli_created = tmp_path / "cli_created.c3d"
    code_create = c3d_metadata_main(
        [
            "--create",
            "-o",
            str(cli_created),
            "--fps",
            "150",
            "--markers",
            "5",
            "--analogs",
            "2",
        ]
    )
    assert code_create == 0
    assert cli_created.is_file()

    # Mode 3: modify
    cli_modified = tmp_path / "cli_mod.c3d"
    code_mod = c3d_metadata_main(
        [
            "-i",
            str(cli_created),
            "-o",
            str(cli_modified),
            "--manufacturer",
            "BiomechLab",
        ]
    )
    assert code_mod == 0
    assert cli_modified.is_file()

    meta_mod = read_c3d_metadata(cli_modified)
    assert meta_mod["manufacturer"]["company"] == "BiomechLab"

    # Error handling: create without output
    code_err = c3d_metadata_main(["--create"])
    assert code_err == 1
