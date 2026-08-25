"""Tests for interp_smooth_core + interp_smooth_split shared pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vaila.interp_smooth_core import (
    align_signals_for_comparison,
    apply_smoothing_1d,
    build_target_time_grid,
    compute_derivatives,
    compute_residual_metrics,
    estimate_sampling_rate,
    first_signal_column,
    is_frame_column_name,
    is_index_column,
    lowess_smooth,
    rebuild_frame_index,
    recommend_butterworth,
    resample_dataframe,
    savgol_smooth,
    validate_butterworth_params,
    validate_time_axis,
)
from vaila.interp_smooth_split import generate_report, process_file
from vaila.interp_smooth_split import savgol_smooth as mod_savgol


def _synth(fs=100.0, seconds=2.0):
    t = np.arange(0, seconds, 1.0 / fs)
    clean = np.sin(2 * np.pi * 2.0 * t)
    noise = 0.15 * np.sin(2 * np.pi * 30.0 * t)
    return t, clean + noise, clean


def test_savgol_smooth_defined_and_runs():
    _, signal, _ = _synth()
    out = savgol_smooth(signal, 7, 3)
    assert out.shape == signal.shape
    assert np.isfinite(out).all()
    # Module re-exports the same helper (fixes NameError)
    assert mod_savgol is savgol_smooth or callable(mod_savgol)


def test_lowess_smooth_runs():
    _, signal, _ = _synth(seconds=1.0)
    out = lowess_smooth(signal, frac=0.3, it=1)
    assert out.shape == signal.shape


def test_butterworth_rejects_cutoff_at_or_above_nyquist():
    errs = validate_butterworth_params(fs=100.0, cutoff=50.0, order=4)
    assert errs
    errs2 = validate_butterworth_params(fs=100.0, cutoff=10.0, order=4)
    assert errs2 == []


def test_butterworth_smoothing_raises_on_invalid_cutoff():
    _, signal, _ = _synth()
    with pytest.raises(ValueError, match="Nyquist"):
        apply_smoothing_1d(signal, "butterworth", {"fs": 100.0, "cutoff": 60.0, "order": 4})


def test_butterworth_reduces_high_freq_noise():
    t, noisy, clean = _synth()
    filtered = apply_smoothing_1d(noisy, "butterworth", {"fs": 100.0, "cutoff": 8.0, "order": 4})
    assert (
        compute_residual_metrics(clean, filtered)["rms"]
        < compute_residual_metrics(clean, noisy)["rms"]
    )


def test_recommend_butterworth_returns_below_nyquist():
    _, signal, _ = _synth()
    rec = recommend_butterworth(signal, fs=100.0)
    assert 0 < rec["recommended_cutoff"] < rec["nyquist"]
    assert "criterion" in rec


def test_validate_time_axis_regular_irregular_invalid():
    t = np.linspace(0, 1, 101)
    info = validate_time_axis(t)
    assert info.status == "regular"
    assert info.estimated_rate_hz == pytest.approx(100.0, rel=1e-3)

    irreg = np.cumsum(np.linspace(0.005, 0.02, 50))
    info2 = validate_time_axis(irreg)
    assert info2.status in {"irregular", "regular"}

    bad = np.array([0.0, 1.0, 1.0, 2.0])
    assert validate_time_axis(bad).status == "invalid"


def test_estimate_sampling_rate_precedence():
    t = np.arange(0, 1, 0.01)
    rate, source, _ = estimate_sampling_rate(t, configured_rate=240.0)
    assert source == "time_column"
    assert rate == pytest.approx(100.0, rel=1e-3)
    rate2, source2, _ = estimate_sampling_rate(None, configured_rate=240.0)
    assert source2 == "configured"
    assert rate2 == 240.0


def test_derivatives_physical_vs_per_sample():
    t = np.linspace(0, 1, 101)
    y = np.sin(2 * np.pi * 2 * t)
    d = compute_derivatives(y, time=t)
    assert d.mode == "physical_time"
    assert d.units_first == "unit/s"
    d2 = compute_derivatives(y)
    assert d2.mode == "per_sample"


def test_resample_down_preserves_duration():
    fs = 100.0
    t = np.arange(0, 1.0 + 1e-12, 1.0 / fs)
    df = pd.DataFrame({"Time": t, "x": np.sin(2 * np.pi * 2 * t), "label": ["A"] * len(t)})
    out, warns = resample_dataframe(
        df,
        time_col="Time",
        original_rate=100.0,
        final_rate=50.0,
        numeric_cols=["x"],
        antialias=True,
    )
    assert abs((out["Time"].iloc[-1] - out["Time"].iloc[0]) - 1.0) < 1e-6
    assert len(out) == len(build_target_time_grid(0.0, 1.0, 50.0))
    assert "label" in out.columns


def test_process_file_savgol_cli_path(tmp_path):
    fs = 50.0
    t = np.arange(0, 1.0, 1.0 / fs)
    df = pd.DataFrame({"Time": t, "x": np.sin(2 * np.pi * 3 * t)})
    src = tmp_path / "sig.csv"
    df.to_csv(src, index=False)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    info = process_file(
        str(src),
        str(out_dir),
        {
            "interp_method": "linear",
            "smooth_method": "savgol",
            "smooth_params": {"window_length": 5, "polyorder": 2},
            "padding": 0.0,
            "max_gap": 0,
            "do_split": False,
            "sample_rate": None,
        },
    )
    assert info is not None
    assert not info.get("error")
    assert info["output_path"]
    assert __import__("os").path.isfile(info["output_path"])


def test_process_file_butterworth_invalid_cutoff_keeps_column(tmp_path):
    t = np.arange(0, 1.0, 0.01)
    df = pd.DataFrame({"Time": t, "x": np.sin(2 * np.pi * 2 * t)})
    src = tmp_path / "sig.csv"
    df.to_csv(src, index=False)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    info = process_file(
        str(src),
        str(out_dir),
        {
            "interp_method": "none",
            "smooth_method": "butterworth",
            "smooth_params": {"fs": 100.0, "cutoff": 80.0, "order": 4},
            "padding": 0.0,
            "max_gap": 0,
            "do_split": False,
            "sample_rate": None,
        },
    )
    # Should not crash; warning recorded when cutoff invalid
    assert info is not None
    out = pd.read_csv(info["output_path"])
    assert "x" in out.columns


def test_resample_dataframe_excludes_index_cols():
    fs = 100.0
    t = np.arange(0, 1.0 + 1e-12, 1.0 / fs)
    frames = np.arange(len(t), dtype=float)
    df = pd.DataFrame({"Time": t, "frame": frames, "x": np.sin(2 * np.pi * 2 * t)})
    out, _warns = resample_dataframe(
        df,
        time_col="Time",
        original_rate=100.0,
        final_rate=50.0,
        numeric_cols=["x"],
        antialias=False,
        exclude_cols=["frame"],
    )
    assert "frame" not in out.columns
    assert len(out) == len(build_target_time_grid(0.0, 1.0, 50.0))


def test_rebuild_frame_index_and_names():
    assert is_frame_column_name("frame")
    assert is_frame_column_name("Frame")
    assert is_index_column("Time")
    assert is_index_column("frame")
    np.testing.assert_array_equal(rebuild_frame_index(5), np.arange(5))


def test_process_file_does_not_smooth_frame_column(tmp_path):
    """A `frame` column past position 0 must stay untouched, not be smoothed."""
    n = 200
    frames = np.arange(n, dtype=float)
    df = pd.DataFrame(
        {
            "Time": np.arange(n) / 240.0,
            "frame": frames,
            "x": np.sin(2 * np.pi * 3 * np.arange(n) / 240.0),
        }
    )
    src = tmp_path / "tf.csv"
    df.to_csv(src, index=False)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    info = process_file(
        str(src),
        str(out_dir),
        {
            "interp_method": "linear",
            "smooth_method": "savgol",
            "smooth_params": {"window_length": 7, "polyorder": 3},
            "padding": 0.0,
            "max_gap": 0,
            "do_split": False,
            "sample_rate": None,
        },
    )
    assert info and not info.get("error")
    out = pd.read_csv(info["output_path"])
    np.testing.assert_allclose(out["frame"].to_numpy(), frames, atol=0)
    assert not np.allclose(out["x"].to_numpy(), df["x"].to_numpy(), atol=0)


@pytest.mark.parametrize("n", [37, 300, 499])
@pytest.mark.parametrize("padding", [5.0, 10.0, 17.0])
def test_padding_preserves_row_count_with_time_column(tmp_path, n, padding):
    """np.arange with a float step used to emit pad_len+1 rows on some sizes."""
    df = pd.DataFrame({"Time": np.arange(n) / 240.0, "x": np.sin(np.arange(n) / 50.0)})
    src = tmp_path / f"pad_{n}_{padding}.csv"
    df.to_csv(src, index=False)
    out_dir = tmp_path / f"out_{n}_{padding}"
    out_dir.mkdir()
    info = process_file(
        str(src),
        str(out_dir),
        {
            "interp_method": "linear",
            "smooth_method": "savgol",
            "smooth_params": {"window_length": 7, "polyorder": 3},
            "padding": padding,
            "max_gap": 0,
            "do_split": False,
            "sample_rate": None,
        },
    )
    assert info and not info.get("error")
    out = pd.read_csv(info["output_path"])
    assert len(out) == n


def test_process_file_resample_frame_first_downsample(tmp_path):
    n = 100
    df = pd.DataFrame({"frame": np.arange(n), "x": np.sin(2 * np.pi * 2 * np.arange(n) / 100.0)})
    src = tmp_path / "frames.csv"
    df.to_csv(src, index=False)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    info = process_file(
        str(src),
        str(out_dir),
        {
            "interp_method": "none",
            "smooth_method": "none",
            "smooth_params": {},
            "padding": 0.0,
            "max_gap": 0,
            "do_split": False,
            "sample_rate": None,
            "resample": True,
            "original_rate": 100.0,
            "final_rate": 50.0,
            "antialias": False,
        },
    )
    assert info and not info.get("error")
    out = pd.read_csv(info["output_path"])
    assert list(out.columns[:2]) == ["frame", "x"]
    assert "Time" not in out.columns
    expected_n = len(build_target_time_grid(0.0, (n - 1) / 100.0, 50.0))
    assert len(out) == expected_n
    np.testing.assert_array_equal(out["frame"].to_numpy(), np.arange(len(out)))


def test_process_file_resample_frame_first_upsample(tmp_path):
    n = 100
    df = pd.DataFrame({"frame": np.arange(n), "x": np.sin(2 * np.pi * 2 * np.arange(n) / 100.0)})
    src = tmp_path / "frames.csv"
    df.to_csv(src, index=False)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    info = process_file(
        str(src),
        str(out_dir),
        {
            "interp_method": "none",
            "smooth_method": "none",
            "smooth_params": {},
            "padding": 0.0,
            "max_gap": 0,
            "do_split": False,
            "sample_rate": None,
            "resample": True,
            "original_rate": 100.0,
            "final_rate": 200.0,
            "antialias": False,
        },
    )
    assert info and not info.get("error")
    out = pd.read_csv(info["output_path"])
    assert "Time" not in out.columns
    assert len(out) > n
    np.testing.assert_array_equal(out["frame"].to_numpy(), np.arange(len(out)))


def test_align_signals_for_comparison_resample_lengths():
    orig = np.sin(2 * np.pi * 2 * np.arange(100) / 100.0)
    proc = np.sin(2 * np.pi * 2 * np.arange(51) / 50.0)
    o, p, note = align_signals_for_comparison(orig, proc, original_rate=100.0, final_rate=50.0)
    assert len(o) == len(p) == 51
    assert "resample_aligned" in note


def test_first_signal_column_skips_frame():
    df = pd.DataFrame({"frame": np.arange(5), "x": np.arange(5, dtype=float)})
    assert first_signal_column(df) == "x"


def test_generate_report_verification_after_resample(tmp_path):
    n = 100
    df = pd.DataFrame({"frame": np.arange(n), "x": np.sin(2 * np.pi * 2 * np.arange(n) / 100.0)})
    src = tmp_path / "frames.csv"
    df.to_csv(src, index=False)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    config = {
        "interp_method": "none",
        "smooth_method": "butterworth",
        "smooth_params": {"fs": 100.0, "cutoff": 8.0, "order": 4},
        "padding": 0.0,
        "max_gap": 0,
        "do_split": False,
        "sample_rate": None,
        "resample": True,
        "original_rate": 100.0,
        "final_rate": 50.0,
        "antialias": False,
    }
    info = process_file(str(src), str(out_dir), config)
    assert info and not info.get("error")
    report_path = generate_report(out_dir, config, [info])
    text = Path(report_path).read_text(encoding="utf-8")
    assert "Error during verification" not in text
    assert "resample_aligned" in text or "Alignment:" in text


def test_cli_help_mentions_resample():
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "vaila/interp_smooth_split.py", "--help"],
        cwd="/home/preto/data/vaila",
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "--resample" in proc.stdout
    assert "--smooth-method" in proc.stdout
    assert "Butterworth" in proc.stdout or "butterworth" in proc.stdout.lower()


def test_upsample_preserves_duration():
    fs = 30.0
    t = np.arange(0, 1.0 + 1e-12, 1.0 / fs)
    df = pd.DataFrame({"Time": t, "x": np.sin(2 * np.pi * 2 * t)})
    out, _warns = resample_dataframe(
        df,
        time_col="Time",
        original_rate=30.0,
        final_rate=60.0,
        numeric_cols=["x"],
        antialias=False,
    )
    assert abs((out["Time"].iloc[-1] - out["Time"].iloc[0]) - 1.0) < 1e-6
    assert len(out) > len(df)


def test_process_file_resample_downsample(tmp_path):
    fs = 100.0
    t = np.arange(0, 1.0 + 1e-12, 1.0 / fs)
    df = pd.DataFrame({"Time": t, "x": np.sin(2 * np.pi * 2 * t)})
    src = tmp_path / "sig.csv"
    df.to_csv(src, index=False)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    info = process_file(
        str(src),
        str(out_dir),
        {
            "interp_method": "none",
            "smooth_method": "none",
            "smooth_params": {},
            "padding": 0.0,
            "max_gap": 0,
            "do_split": False,
            "sample_rate": None,
            "resample": True,
            "original_rate": 100.0,
            "final_rate": 50.0,
            "antialias": True,
        },
    )
    assert info and not info.get("error")
    out = pd.read_csv(info["output_path"])
    assert abs(float(out["Time"].iloc[-1]) - float(out["Time"].iloc[0]) - 1.0) < 1e-5
    assert len(out) < len(df)


def test_gui_cli_config_parity_via_shared_smoothing(tmp_path):
    """Same config dict yields identical numeric output (shared process_file path)."""
    fs = 100.0
    t = np.arange(0, 2.0, 1.0 / fs)
    noisy = np.sin(2 * np.pi * 2 * t) + 0.1 * np.sin(2 * np.pi * 25 * t)
    df = pd.DataFrame({"Time": t, "x": noisy})
    src = tmp_path / "sig.csv"
    df.to_csv(src, index=False)

    config = {
        "interp_method": "linear",
        "smooth_method": "butterworth",
        "smooth_params": {"fs": 100.0, "cutoff": 8.0, "order": 4},
        "padding": 0.0,
        "max_gap": 0,
        "do_split": False,
        "sample_rate": None,
    }
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    out_a.mkdir()
    out_b.mkdir()
    info_a = process_file(str(src), str(out_a), config)
    info_b = process_file(str(src), str(out_b), dict(config))
    a = pd.read_csv(info_a["output_path"])
    b = pd.read_csv(info_b["output_path"])
    np.testing.assert_allclose(a["x"].to_numpy(), b["x"].to_numpy(), rtol=0, atol=0)


def test_toml_round_trip_includes_resample(tmp_path):
    from vaila.interp_smooth_split import load_smooth_config_for_analysis, save_smooth_config_toml

    cfg = {
        "interp_method": "linear",
        "smooth_method": "butterworth",
        "smooth_params": {"fs": 100.0, "cutoff": 10.0, "order": 4},
        "padding": 5.0,
        "max_gap": 30,
        "do_split": False,
        "sample_rate": 240.0,
        "resample": True,
        "original_rate": 100.0,
        "final_rate": 50.0,
        "antialias": True,
        "antialias_cutoff": None,
    }
    path = tmp_path / "smooth_config.toml"
    save_smooth_config_toml(cfg, str(path))
    loaded = load_smooth_config_for_analysis(str(path))
    assert loaded["smooth_method"] == "butterworth"
    assert loaded["smooth_params"]["order"] == 4
    assert loaded["resample"] is True
    assert loaded["original_rate"] == 100.0
    assert loaded["final_rate"] == 50.0
    assert loaded["sample_rate"] == 240.0


def test_cli_flag_overrides_toml(tmp_path):
    import subprocess
    import sys

    from vaila.interp_smooth_split import save_smooth_config_toml

    fs = 50.0
    t = np.arange(0, 0.5, 1.0 / fs)
    df = pd.DataFrame({"Time": t, "x": np.sin(2 * np.pi * 3 * t)})
    src_dir = tmp_path / "in"
    src_dir.mkdir()
    df.to_csv(src_dir / "sig.csv", index=False)
    toml_path = tmp_path / "cfg.toml"
    save_smooth_config_toml(
        {
            "interp_method": "none",
            "smooth_method": "none",
            "smooth_params": {},
            "padding": 0.0,
            "max_gap": 0,
            "do_split": False,
            "sample_rate": None,
        },
        str(toml_path),
    )
    out = tmp_path / "out"
    proc = subprocess.run(
        [
            sys.executable,
            "vaila/interp_smooth_split.py",
            "-i",
            str(src_dir),
            "-o",
            str(out),
            "-c",
            str(toml_path),
            "--smooth-method",
            "savgol",
            "--window-length",
            "5",
            "--polyorder",
            "2",
        ],
        cwd="/home/preto/data/vaila",
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    outs = list(out.glob("*_savgol.csv"))
    assert outs, "CLI --smooth-method should override TOML none"
