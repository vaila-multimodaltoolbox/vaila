"""Regression tests for vailaplot2d time-scatter X-axis resolution."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vaila import vailaplot2d as plot2d


def test_resolve_x_single_column_force_csv():
    """Force-plate CSVs with only fz_N must use sample index, not skip the series."""
    df = pd.DataFrame({"fz_N": [-610.2, -611.1, -612.0]})
    out, x_col, x_label = plot2d.resolve_time_scatter_x_axis(df)
    assert x_col == "Time_Index"
    assert x_label == "Sample Index"
    assert list(out["Time_Index"]) == [1, 2, 3]
    assert "fz_N" in out.columns


def test_resolve_x_prefers_valid_time_column():
    df = pd.DataFrame({"Time": [0.0, 0.01, 0.02], "fz_N": [1.0, 2.0, 3.0]})
    out, x_col, x_label = plot2d.resolve_time_scatter_x_axis(df)
    assert x_col == "Time"
    assert x_label == "Time"
    assert out is df or list(out.columns) == list(df.columns)


def test_resolve_x_rejects_degenerate_time():
    df = pd.DataFrame({"Time": [0, 1, 0, 1], "fz_N": [1.0, 2.0, 3.0, 4.0]})
    out, x_col, x_label = plot2d.resolve_time_scatter_x_axis(df)
    assert x_col == "Time_Index"
    assert x_label == "Sample Index"
    assert "fz_N" in out.columns


def test_plot_time_scatter_single_header_no_zero_plots(monkeypatch, tmp_path):
    """End-to-end: one file, one header, no Time column → at least one line plotted."""
    csv_path = tmp_path / "s01_fp2_r_newton.csv"
    pd.DataFrame({"fz_N": np.linspace(-600, -650, 20)}).to_csv(csv_path, index=False)

    plot2d.selected_files = [str(csv_path)]
    plot2d.selected_headers = ["fz_N"]
    plot2d.loaded_data_cache = {}

    warnings: list[tuple] = []
    monkeypatch.setattr(plot2d.messagebox, "showwarning", lambda *a, **k: warnings.append(a))
    monkeypatch.setattr(plt, "show", lambda: None)

    plt.close("all")
    plot2d.plot_time_scatter()

    assert warnings == [], f"unexpected warning: {warnings}"
    fig = plt.gcf()
    axes = fig.get_axes()
    assert axes, "expected an axes after plot"
    lines = axes[0].get_lines()
    assert len(lines) >= 1
    assert len(lines[0].get_ydata()) == 20
    plt.close("all")
