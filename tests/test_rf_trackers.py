"""Smoke tests for vaila.rf_trackers (Roboflow trackers + YOLO bridge)."""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from vaila.rf_trackers import (
    TrackerName,
    _cmc_method_from_ui,
    _write_vaila_per_id_tracking_csvs,
    build_arg_parser,
    build_tracker,
)

_REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "name",
    ["sort", "bytetrack", "ocsort", "botsort"],
)
def test_build_tracker_instantiation(name: TrackerName) -> None:
    tr = build_tracker(name, enable_cmc=False) if name == "botsort" else build_tracker(name)
    assert tr is not None
    assert hasattr(tr, "update")


def test_cli_help_lists_options() -> None:
    r = subprocess.run(
        [sys.executable, "-m", "vaila.rf_trackers", "--help"],
        capture_output=True,
        text=True,
        check=False,
        cwd=_REPO,
        env={**os.environ, "PYTHONPATH": str(_REPO)},
    )
    assert r.returncode == 0
    out = r.stdout + r.stderr
    assert "BRA_KOR_234113" in out
    assert "--input" in out or "-i" in out
    assert "--quiet" in out or "-q" in out


def test_cmc_method_from_ui_normalizes_sparse_opt_flow() -> None:
    assert _cmc_method_from_ui("sparseOptFlow") == "sparseOptFlow"


def test_build_arg_parser_default_is_gui_mode() -> None:
    p = build_arg_parser()
    ns = p.parse_args([])
    assert ns.input is None


def test_default_weights_is_yolo26x() -> None:
    ns = build_arg_parser().parse_args([])
    assert ns.weights.name == "yolo26x.pt"


def test_argparse_quiet() -> None:
    ns = build_arg_parser().parse_args(["-i", __file__, "-q"])
    assert ns.quiet is True


def test_build_arg_parser_cli_requires_resolved_paths() -> None:
    ns = build_arg_parser().parse_args(["-i", __file__, "-w", __file__, "--tracker", "sort"])
    assert ns.input is not None
    assert ns.tracker == "sort"


def test_write_vaila_per_id_tracking_csvs_and_combined(tmp_path: Path) -> None:
    """Synthetic detections -> yolov26-style per-ID CSVs + all_id_detection.csv."""
    n_frames = 5
    rows_out = [
        [0, 1, 10.0, 20.0, 50.0, 80.0, 0.9, 0, "person"],
        [2, 1, 11.0, 21.0, 51.0, 81.0, 0.8, 0, "person"],
        [1, 2, 5.0, 5.0, 30.0, 40.0, 0.7, 2, "car"],
        [3, 2, 6.0, 6.0, 31.0, 41.0, 0.95, 2, "car"],
    ]
    combined = _write_vaila_per_id_tracking_csvs(tmp_path, n_frames, rows_out, show_progress=False)
    assert combined is not None
    assert combined.name == "all_id_detection.csv"

    p_person = tmp_path / "person_id_01.csv"
    p_car = tmp_path / "car_id_02.csv"
    assert p_person.is_file()
    assert p_car.is_file()

    exp_cols = [
        "Frame",
        "Tracker ID",
        "Label",
        "X_min",
        "Y_min",
        "X_max",
        "Y_max",
        "Confidence",
        "Color_R",
        "Color_G",
        "Color_B",
    ]
    df_p = pd.read_csv(p_person)
    assert list(df_p.columns) == exp_cols
    assert len(df_p) == n_frames
    row1 = df_p.loc[df_p["Frame"] == 1].iloc[0]
    assert math.isnan(row1["X_min"])
    assert row1["Tracker ID"] == 1
    assert row1["Label"] == "person"

    df_all = pd.read_csv(combined)
    assert "Frame" in df_all.columns
    assert "X_min_person_id_01" in df_all.columns
    assert "X_min_car_id_02" in df_all.columns
    assert len(df_all) == n_frames


def test_write_vaila_per_id_tracking_csvs_empty_returns_none(tmp_path: Path) -> None:
    assert _write_vaila_per_id_tracking_csvs(tmp_path, 10, [], show_progress=False) is None
    assert (
        _write_vaila_per_id_tracking_csvs(
            tmp_path, 0, [[0, 1, 0, 0, 1, 1, 1.0, 0, "a"]], show_progress=False
        )
        is None
    )
