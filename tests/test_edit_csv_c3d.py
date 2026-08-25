"""Tests for vaila/edit_csv_c3d.py — the Edit CSV/C3D directory pipeline.

Covers the headless path only (no GUI/Tk): CSV identity/reorder, C3D
round-trip fidelity (labels, rate, units), column subset/reorder, occlusion
residual preservation, analog-channel preservation, the GUI->CLI mirror
banner, and that the headless code path never touches `tkinter.Tk`.
"""

import numpy as np
import pandas as pd
import pytest

ezc3d = pytest.importorskip("ezc3d")

try:
    from vaila.cli_highlight import print_gui_cli_mirror
    from vaila.edit_csv_c3d import _headless_process, main
    from vaila.readcsv_export import auto_create_c3d_from_csv
except ImportError:  # standalone execution
    from cli_highlight import print_gui_cli_mirror  # ty: ignore[unresolved-import]
    from edit_csv_c3d import _headless_process, main  # ty: ignore[unresolved-import]
    from readcsv_export import auto_create_c3d_from_csv  # ty: ignore[unresolved-import]


def _points_df(n_frames=6, markers=("p1", "p2", "p3")):
    data = {"frame": np.arange(n_frames, dtype=float)}
    for i, m in enumerate(markers, start=1):
        for axis, off in zip(("X", "Y", "Z"), (1.0, 2.0, 3.0), strict=True):
            data[f"{m}_{axis}"] = float(i) + off + np.arange(n_frames, dtype=float)
    return pd.DataFrame(data)


def test_csv_identity_and_reorder(tmp_path):
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    csv_path = input_dir / "trial.csv"
    df = pd.DataFrame({"frame": [0, 1, 2], "a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    df.to_csv(csv_path, index=False)
    original_bytes = csv_path.read_bytes()

    output_dir = tmp_path / "out"
    written = _headless_process(str(input_dir), str(output_dir), columns=["frame", "b", "a"])

    assert len(written) == 1
    out_df = pd.read_csv(written[0])
    assert list(out_df.columns) == ["frame", "b", "a"]
    np.testing.assert_allclose(out_df["a"], df["a"])
    np.testing.assert_allclose(out_df["b"], df["b"])
    assert csv_path.read_bytes() == original_bytes, "source CSV must stay untouched"


def test_c3d_identity_round_trip(tmp_path):
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    df = _points_df()
    c3d_path = input_dir / "take.c3d"
    auto_create_c3d_from_csv(df, str(c3d_path), point_rate=100.0, point_units="mm")
    original_bytes = c3d_path.read_bytes()

    output_dir = tmp_path / "out"
    written = _headless_process(str(input_dir), str(output_dir), columns=None)

    assert len(written) == 1
    out_c3d = ezc3d.c3d(written[0])
    src_c3d = ezc3d.c3d(str(c3d_path))

    assert out_c3d["parameters"]["POINT"]["LABELS"]["value"] == ["p1", "p2", "p3"]
    assert out_c3d["parameters"]["POINT"]["RATE"]["value"] == [100.0]
    assert out_c3d["parameters"]["POINT"]["UNITS"]["value"] == ["mm"]
    np.testing.assert_allclose(
        out_c3d["data"]["points"][:3], src_c3d["data"]["points"][:3], atol=1e-5
    )
    assert c3d_path.read_bytes() == original_bytes, "source C3D must stay untouched"


def test_c3d_columns_subset_reorder(tmp_path):
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    df = _points_df(markers=("p1", "p2", "p3"))
    c3d_path = input_dir / "take.c3d"
    auto_create_c3d_from_csv(df, str(c3d_path), point_rate=100.0)

    output_dir = tmp_path / "out"
    columns = ["Time", "p3_X", "p3_Y", "p3_Z", "p1_X", "p1_Y", "p1_Z"]
    written = _headless_process(str(input_dir), str(output_dir), columns=columns)

    out_c3d = ezc3d.c3d(written[0])
    assert out_c3d["parameters"]["POINT"]["LABELS"]["value"] == ["p3", "p1"]


def test_occlusion_residuals_preserved_through_round_trip(tmp_path):
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    df = _points_df()
    df.loc[2:3, ["p2_X", "p2_Y", "p2_Z"]] = np.nan
    c3d_path = input_dir / "take.c3d"
    auto_create_c3d_from_csv(df, str(c3d_path), point_rate=100.0)

    output_dir = tmp_path / "out"
    written = _headless_process(str(input_dir), str(output_dir), columns=None)

    out_c3d = ezc3d.c3d(written[0])
    residuals = out_c3d["data"]["meta_points"]["residuals"]
    invalid = residuals[0] < 0
    assert invalid.sum() == 2, "exactly the two occluded samples must round-trip as invalid"
    assert invalid[1, 2] and invalid[1, 3]

    points = out_c3d["data"]["points"]
    valid = residuals[0] >= 0
    at_origin = (np.abs(points[:3]) < 1e-9).all(axis=0)
    assert not at_origin[valid].any(), "occluded marker must not leak in as a valid origin point"


def test_analog_preserved_when_present(tmp_path):
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    df = _points_df()
    analog_df = pd.DataFrame(
        {
            "Time": [i / 1000.0 for i in range(60)],
            "EMG1": np.sin(np.linspace(0, 6.28, 60)),
        }
    )
    c3d_path = input_dir / "take.c3d"
    auto_create_c3d_from_csv(
        df, str(c3d_path), analog_df=analog_df, point_rate=100.0, analog_rate=1000.0
    )

    output_dir = tmp_path / "out"
    written = _headless_process(str(input_dir), str(output_dir), columns=None)

    out_c3d = ezc3d.c3d(written[0])
    assert out_c3d["parameters"]["ANALOG"]["LABELS"]["value"] == ["EMG1"]
    assert out_c3d["parameters"]["ANALOG"]["RATE"]["value"] == [1000.0]
    assert out_c3d["data"]["analogs"].shape[-1] == 60


def test_gui_cli_mirror_contains_input_and_output_flags(capsys):
    print_gui_cli_mirror(
        "vaila/edit_csv_c3d",
        ["uv", "run", "vaila/edit_csv_c3d.py", "-i", "/some/input", "-o", "/some/output"],
    )
    out = capsys.readouterr().out
    assert "uv run vaila/edit_csv_c3d.py" in out
    assert "-i" in out
    assert "-o" in out
    assert "/some/input" in out
    assert "/some/output" in out


def test_headless_path_never_touches_tk(tmp_path, monkeypatch):
    import sys
    import tkinter

    def _boom(*args, **kwargs):
        raise AssertionError("tkinter.Tk() must not be instantiated in the headless path")

    monkeypatch.setattr(tkinter, "Tk", _boom)

    input_dir = tmp_path / "in"
    input_dir.mkdir()
    df = pd.DataFrame({"frame": [0, 1], "a": [1.0, 2.0]})
    (input_dir / "trial.csv").write_text(df.to_csv(index=False))

    output_dir = tmp_path / "out"
    written = _headless_process(str(input_dir), str(output_dir), columns=None)
    assert written

    argv = [
        "edit_csv_c3d.py",
        "-i",
        str(input_dir),
        "-o",
        str(tmp_path / "out2"),
        "--identity",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    main()
