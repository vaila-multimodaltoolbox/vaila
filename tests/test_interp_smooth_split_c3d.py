"""C3D roundtrip tests for interp_smooth_split (C3D→CSV→process→C3D)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

ezc3d = pytest.importorskip("ezc3d")

from vaila.interp_smooth_split import process_c3d_file, run_batch  # noqa: E402
from vaila.readcsv_export import auto_create_c3d_from_csv  # noqa: E402


def _points_df(n_frames=20, markers=("p1", "p2", "p3"), fs=100.0):
    t = np.arange(n_frames, dtype=float) / fs
    data = {"Time": t}
    for i, m in enumerate(markers, start=1):
        for axis, off in zip(("X", "Y", "Z"), (1.0, 2.0, 3.0), strict=True):
            data[f"{m}_{axis}"] = float(i) + off + 0.1 * np.sin(2 * np.pi * 2.0 * t + i)
    return pd.DataFrame(data)


def _noop_config(**overrides):
    cfg = {
        "interp_method": "none",
        "smooth_method": "none",
        "smooth_params": {},
        "padding": 0.0,
        "max_gap": 0,
        "do_split": False,
        "sample_rate": None,
        "resample": False,
    }
    cfg.update(overrides)
    return cfg


def test_c3d_identity_round_trip(tmp_path):
    df = _points_df()
    src = tmp_path / "take.c3d"
    auto_create_c3d_from_csv(df, str(src), point_rate=100.0, point_units="mm")
    original_bytes = src.read_bytes()
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    info = process_c3d_file(str(src), str(out_dir), _noop_config())
    assert info is not None
    assert not info.get("error"), info.get("warnings")
    assert info["output_path"].endswith(".c3d")
    assert src.read_bytes() == original_bytes

    out_c3d = ezc3d.c3d(info["output_path"])
    src_c3d = ezc3d.c3d(str(src))
    assert out_c3d["parameters"]["POINT"]["LABELS"]["value"] == ["p1", "p2", "p3"]
    assert out_c3d["parameters"]["POINT"]["RATE"]["value"] == [100.0]
    assert out_c3d["parameters"]["POINT"]["UNITS"]["value"] == ["mm"]
    np.testing.assert_allclose(
        out_c3d["data"]["points"][:3], src_c3d["data"]["points"][:3], atol=1e-5
    )


def test_c3d_occlusion_residuals_preserved(tmp_path):
    df = _points_df()
    df.loc[2:3, ["p2_X", "p2_Y", "p2_Z"]] = np.nan
    src = tmp_path / "occ.c3d"
    auto_create_c3d_from_csv(df, str(src), point_rate=100.0)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    info = process_c3d_file(str(src), str(out_dir), _noop_config())
    assert not info.get("error"), info.get("warnings")
    out_c3d = ezc3d.c3d(info["output_path"])
    residuals = out_c3d["data"]["meta_points"]["residuals"]
    invalid = residuals[0] < 0
    assert invalid.sum() == 2
    points = out_c3d["data"]["points"]
    valid = residuals[0] >= 0
    at_origin = (np.abs(points[:3]) < 1e-9).all(axis=0)
    assert not at_origin[valid].any()


def test_c3d_analog_preserved_when_present(tmp_path):
    df = _points_df(n_frames=6)
    analog_df = pd.DataFrame(
        {
            "Time": [i / 1000.0 for i in range(60)],
            "EMG1": np.sin(np.linspace(0, 6.28, 60)),
        }
    )
    src = tmp_path / "with_analog.c3d"
    auto_create_c3d_from_csv(
        df, str(src), analog_df=analog_df, point_rate=100.0, analog_rate=1000.0
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    info = process_c3d_file(str(src), str(out_dir), _noop_config())
    assert not info.get("error"), info.get("warnings")
    out_c3d = ezc3d.c3d(info["output_path"])
    assert out_c3d["parameters"]["ANALOG"]["LABELS"]["value"] == ["EMG1"]
    assert out_c3d["parameters"]["ANALOG"]["RATE"]["value"] == [1000.0]
    assert out_c3d["data"]["analogs"].shape[-1] == 60


def test_c3d_savgol_changes_points(tmp_path):
    df = _points_df(n_frames=50)
    src = tmp_path / "smooth.c3d"
    auto_create_c3d_from_csv(df, str(src), point_rate=100.0)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    info = process_c3d_file(
        str(src),
        str(out_dir),
        _noop_config(
            smooth_method="savgol",
            smooth_params={"window_length": 5, "polyorder": 2},
        ),
    )
    assert not info.get("error"), info.get("warnings")
    out_c3d = ezc3d.c3d(info["output_path"])
    src_c3d = ezc3d.c3d(str(src))
    assert out_c3d["parameters"]["POINT"]["LABELS"]["value"] == ["p1", "p2", "p3"]
    assert out_c3d["parameters"]["POINT"]["RATE"]["value"] == [100.0]
    assert np.isfinite(out_c3d["data"]["points"][:3]).all()
    assert not np.allclose(out_c3d["data"]["points"][:3], src_c3d["data"]["points"][:3], atol=1e-8)


def test_c3d_split_writes_two_c3d(tmp_path):
    df = _points_df(n_frames=20)
    src = tmp_path / "split_me.c3d"
    auto_create_c3d_from_csv(df, str(src), point_rate=100.0)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    info = process_c3d_file(str(src), str(out_dir), _noop_config(do_split=True))
    assert not info.get("error"), info.get("warnings")
    assert info["output_part1_path"].endswith(".c3d")
    assert info["output_part2_path"].endswith(".c3d")
    assert info["part1_size"] == 10
    assert info["part2_size"] == 10
    c1 = ezc3d.c3d(info["output_part1_path"])
    c2 = ezc3d.c3d(info["output_part2_path"])
    assert c1["data"]["points"].shape[-1] == 10
    assert c2["data"]["points"].shape[-1] == 10


def test_run_batch_mixed_csv_and_c3d(tmp_path):
    import os

    input_dir = tmp_path / "in"
    input_dir.mkdir()
    csv_df = pd.DataFrame({"Time": [0.0, 0.01, 0.02], "x": [1.0, 2.0, 3.0]})
    (input_dir / "sig.csv").write_text(csv_df.to_csv(index=False))
    auto_create_c3d_from_csv(_points_df(n_frames=10), str(input_dir / "take.c3d"), point_rate=100.0)
    out_dir = tmp_path / "out"
    dest, files, _report = run_batch(
        str(input_dir), _noop_config(), dest_dir=str(out_dir), use_messagebox=False
    )
    assert dest == str(out_dir)
    assert len([f for f in files if not f.get("error")]) == 2
    kinds = {os.path.splitext(f["output_path"])[1].lower() for f in files if f.get("output_path")}
    assert ".csv" in kinds
    assert ".c3d" in kinds


def test_cli_help_mentions_c3d():
    import subprocess
    import sys
    from pathlib import Path

    proc = subprocess.run(
        [sys.executable, "vaila/interp_smooth_split.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert proc.returncode == 0
    assert "C3D" in proc.stdout or "c3d" in proc.stdout


def test_headless_c3d_never_touches_tk(tmp_path, monkeypatch):
    import tkinter

    def _boom(*_a, **_k):
        raise AssertionError("tkinter.Tk() must not be instantiated in the headless path")

    monkeypatch.setattr(tkinter, "Tk", _boom)
    src = tmp_path / "take.c3d"
    auto_create_c3d_from_csv(_points_df(n_frames=8), str(src), point_rate=100.0)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    info = process_c3d_file(str(src), str(out_dir), _noop_config())
    assert not info.get("error")
