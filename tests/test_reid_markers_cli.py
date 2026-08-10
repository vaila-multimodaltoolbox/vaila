"""Headless CLI smoke tests for `python -m vaila.reid_markers`.

Guards against the exact failure class this repo has hit before with GUI
tools invoked headlessly (rec3d.py/rec2d.py hanging on an unguarded
messagebox call, see docs/sessions/...) -- these tests actually spawn the
real CLI as a subprocess, under a timeout, and assert a clean exit rather
than importing functions in-process.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


def _write_mini_bbox_wide_slot_csv(path: Path) -> None:
    n = 5
    df = pd.DataFrame(
        {
            "Frame": range(n),
            "Tracker ID_person_id_01": [1] * n,
            "Label_person_id_01": ["person"] * n,
            "X_min_person_id_01": [10 + i for i in range(n)],
            "Y_min_person_id_01": [10 + i for i in range(n)],
            "X_max_person_id_01": [30 + i for i in range(n)],
            "Y_max_person_id_01": [30 + i for i in range(n)],
            "Confidence_person_id_01": [0.9] * n,
            "Tracker ID_person_id_02": [2] * n,
            "Label_person_id_02": ["person"] * n,
            "X_min_person_id_02": [500 + i for i in range(n)],
            "Y_min_person_id_02": [500 + i for i in range(n)],
            "X_max_person_id_02": [520 + i for i in range(n)],
            "Y_max_person_id_02": [520 + i for i in range(n)],
            "Confidence_person_id_02": [0.8] * n,
        }
    )
    df.to_csv(path, index=False)


def test_cli_runs_headless_and_exits_zero(tmp_path: Path) -> None:
    input_csv = tmp_path / "all_id_detection.csv"
    _write_mini_bbox_wide_slot_csv(input_csv)
    out_dir = tmp_path / "out"

    proc = subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            "vaila.reid_markers",
            "--input",
            str(input_csv),
            "--max-ids",
            "2",
            "--output-dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        timeout=60,  # hard guard -- a hang here must fail the test, not the suite
        cwd=str(Path(__file__).resolve().parents[1]),
    )

    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert "TclError" not in proc.stderr
    normalized_stdout = " ".join(proc.stdout.split())
    assert "raw_ids=2 -> stable_ids=2" in normalized_stdout
    produced = list(out_dir.glob("*.csv"))
    assert len(produced) == 1
    out_df = pd.read_csv(produced[0])
    assert "X_min_person_id_01" in out_df.columns
    assert "X_min_person_id_02" in out_df.columns


def test_cli_missing_input_file_exits_nonzero_without_hanging(tmp_path: Path) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            "vaila.reid_markers",
            "--input",
            str(tmp_path / "does_not_exist.csv"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert proc.returncode == 2
    assert "not found" in proc.stdout.lower()


def test_cli_unsupported_schema_exits_nonzero_with_actionable_message(tmp_path: Path) -> None:
    input_csv = tmp_path / "unsupported.csv"
    pd.DataFrame({"foo": [1, 2], "bar": [3, 4]}).to_csv(input_csv, index=False)

    proc = subprocess.run(
        [sys.executable, "-u", "-m", "vaila.reid_markers", "--input", str(input_csv)],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert proc.returncode == 1
    assert "Could not auto-detect" in proc.stdout


def test_cli_auto_estimates_max_ids_when_omitted(tmp_path: Path) -> None:
    input_csv = tmp_path / "all_id_detection.csv"
    _write_mini_bbox_wide_slot_csv(input_csv)
    out_dir = tmp_path / "out"

    proc = subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            "vaila.reid_markers",
            "--input",
            str(input_csv),
            "--output-dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    # 2 raw ids, always co-occurring -> auto-estimated peak concurrency is 2.
    normalized_stdout = " ".join(proc.stdout.split())
    assert "(max_ids=2)" in normalized_stdout
