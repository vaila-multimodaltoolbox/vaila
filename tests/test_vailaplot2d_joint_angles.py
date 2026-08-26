"""Tests for vailaplot2d joint-angles time-series helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from vaila.vailaplot2d import (
    is_joint_angles_dataframe,
    load_joint_angles_series,
    plot_joint_angles_time,
    prefer_joint_name,
)

REF_CSV = Path(
    "/home/preto/data/sep_runcod_01072026/REC3D_COD/jessica/vaila_rec3d_out/"
    "vaila_rec3d_20260826_121305/rec3d_20260826_121305_joint_angles.csv"
)


def _synth(n=20, person_id=8, joint_name="left-knee", fs=100.0):
    frames = np.arange(n, dtype=float)
    t = frames / fs
    return pd.DataFrame(
        {
            "frame": frames,
            "person_id": person_id,
            "joint_idx": 3,
            "joint_name": joint_name,
            "parent_idx": 2,
            "euler_x_deg": np.sin(2 * np.pi * 1.0 * t),
            "euler_y_deg": 0.1 * np.cos(2 * np.pi * 1.0 * t),
            "euler_z_deg": np.full(n, 2.5),
            "quat_w": 1.0,
            "quat_x": 0.0,
            "quat_y": 0.0,
            "quat_z": 0.0,
        }
    )


def test_is_joint_angles_dataframe():
    assert is_joint_angles_dataframe(_synth())
    assert not is_joint_angles_dataframe(pd.DataFrame({"Time": [0, 1], "x": [1, 2]}))


def test_prefer_joint_name_knees_first():
    assert prefer_joint_name(["neck", "right-knee", "left-hip"]) == "right-knee"
    assert prefer_joint_name(["neck", "left-knee", "right-knee"]) == "left-knee"
    assert prefer_joint_name(["a", "b"]) == "a"


def test_load_joint_angles_series_sine_and_fs():
    df = _synth(n=50)
    t, series = load_joint_angles_series(df, person_id=8, joint_name="left-knee", fs=100.0)
    assert len(t) == 50
    assert len(series["euler_x_deg"]) == 50
    np.testing.assert_allclose(t, np.arange(50) / 100.0)
    np.testing.assert_allclose(series["euler_x_deg"], df["euler_x_deg"].to_numpy())


def test_load_joint_angles_series_missing_raises():
    df = _synth()
    with pytest.raises(ValueError, match="No rows"):
        load_joint_angles_series(df, person_id=99, joint_name="left-knee")
    with pytest.raises(ValueError, match="Not a joint-angles"):
        load_joint_angles_series(pd.DataFrame({"a": [1]}), person_id=1, joint_name="x")


def test_load_preserves_nan_gaps():
    df = _synth(n=10)
    df.loc[3:4, "euler_x_deg"] = np.nan
    _t, series = load_joint_angles_series(df, person_id=8, joint_name="left-knee")
    assert np.isnan(series["euler_x_deg"][3])
    assert np.isnan(series["euler_x_deg"][4])
    assert np.isfinite(series["euler_x_deg"][0])


def test_plot_joint_angles_time_agg():
    df = _synth()
    fig, ax, t, series = plot_joint_angles_time(
        df, person_id=8, joint_name="left-knee", fs=None, show=False
    )
    assert len(t) == 20
    assert ax.get_xlabel() == "Frame"
    assert "deg" in ax.get_ylabel().lower()
    legends = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "Flexion/Extension" in legends
    assert "Abduction/Adduction" in legends
    assert "Internal/External Rotation" in legends
    plt.close(fig)


@pytest.mark.skipif(not REF_CSV.is_file(), reason="external jessica joint_angles CSV absent")
def test_optional_external_left_knee_person8():
    df = pd.read_csv(REF_CSV)
    t, series = load_joint_angles_series(df, person_id=8, joint_name="left-knee")
    assert len(t) == len(series["euler_x_deg"]) > 0
    assert np.isfinite(series["euler_x_deg"]).any()
