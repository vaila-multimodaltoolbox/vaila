"""End-to-end check of ``vaila.sit2stand`` on the shipped force-plate sample.

``tests/Force_Plate/sit2stand/sit2stand1.csv`` is a 1.87 s recording (2000 Hz, column
``Force.Fz3``) analysed with ``config_sit2stand.toml``. The test runs the CLI batch path
into a temporary directory and locks the force extrema read from the sample, so a change
in loading, column selection or filtering is detected.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from vaila.sit2stand import run_cli_mode

SAMPLE_DIR = Path(__file__).resolve().parent / "Force_Plate" / "sit2stand"
SAMPLE_CSV = SAMPLE_DIR / "sit2stand1.csv"
SAMPLE_TOML = SAMPLE_DIR / "config_sit2stand.toml"


@pytest.fixture
def batch_summary(tmp_path, monkeypatch):
    if not SAMPLE_CSV.exists() or not SAMPLE_TOML.exists():
        pytest.skip(f"sit2stand sample not found in {SAMPLE_DIR}")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    work = tmp_path / "in"
    work.mkdir()
    shutil.copy(SAMPLE_CSV, work / SAMPLE_CSV.name)
    out = tmp_path / "out"
    run_cli_mode([str(SAMPLE_TOML), str(work), str(out)])
    summary = out / "sit2stand_results" / "batch_analysis_summary.csv"
    assert summary.exists(), "batch summary CSV was not written"
    return pd.read_csv(summary), out


def test_sample_runs_and_writes_reports(batch_summary):
    df, out = batch_summary
    assert len(df) == 1
    assert df.iloc[0]["filename"] == "sit2stand1.csv"
    per_file = out / "sit2stand_results" / "sit2stand1"
    for name in (
        "sit2stand1_analysis_report.txt",
        "sit2stand1_analysis_data.csv",
        "sit2stand1_force_plot.png",
    ):
        assert (per_file / name).exists(), f"missing {name}"


def test_sample_force_extrema_and_duration(batch_summary):
    row = batch_summary[0].iloc[0]
    assert row["duration_s"] == pytest.approx(1.8695, abs=1e-3)
    assert row["sampling_frequency_Hz"] == pytest.approx(2000.0)
    assert row["max_force_N"] == pytest.approx(149.16, abs=0.05)
    assert row["min_force_N"] == pytest.approx(57.67, abs=0.05)
    assert row["num_phases"] >= 1
