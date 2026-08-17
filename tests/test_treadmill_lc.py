from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import matplotlib

# Lock Agg before importing treadmill_lc so pyplot never binds QtAgg/PySide6.
matplotlib.use("Agg")

import numpy as np
import pandas as pd

import vaila.treadmill_lc as lct
from vaila.treadmill_lc import (
    adjustment_metadata_to_interval_specs,
    analyze_spectrum_filt,
    apply_adjustment_intervals,
    apply_adjustment_metadata_as_nan,
    apply_filter,
    apply_rbf_interp,
    calculate_cop_system,
    calibration_center_slice,
    canonical_trial_filename,
    deduplicate_trial_files,
    detect_steps,
    discover_calibration_and_borg,
    find_adjustment_metadata_file,
    get_default_interp_config,
    get_group_weight_from_borg,
    is_calibration_file,
    is_trial_file,
    load_adjustment_metadata,
    load_data,
    load_filter_config,
    load_interp_config,
    merge_intervals,
    normalize_adjustment_mode,
    normalize_analysis_window_points,
    plot_trial_figures,
    preprocess_file_interp,
    read_calibration_cells,
    reset_times,
    save_adjustment_metadata,
    save_interp_config,
    strikeattr,
)


def test_merge_intervals():
    # Empty list
    assert merge_intervals([]) == []

    # Overlapping and adjacent intervals
    intervals = [(10, 20), (15, 25), (30, 40), (25, 30)]
    # (10, 20) and (15, 25) overlap -> (10, 25)
    # (25, 30) touches (10, 25) because current[0] (25) <= last[1] (25) -> (10, 30)
    # (30, 40) touches (10, 30) because current[0] (30) <= last[1] (30) -> (10, 40)
    assert merge_intervals(intervals) == [(10, 40)]

    # Non-overlapping intervals
    intervals = [(10, 20), (30, 40), (50, 60)]
    assert merge_intervals(intervals) == [(10, 20), (30, 40), (50, 60)]


def test_reset_times():
    # Empty time array
    np.testing.assert_array_equal(reset_times(np.array([])), np.array([]))

    # Regular time array with gap
    # 0.0, 0.1, 0.2, 0.5, 0.6
    t_clean = np.array([0.0, 0.1, 0.2, 0.5, 0.6])
    t_reset = reset_times(t_clean)
    # Median dt should be 0.1
    expected = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    np.testing.assert_allclose(t_reset, expected)


def test_interp_config_save_load(tmp_path):
    config_file = tmp_path / "interp_config.toml"
    default_config = get_default_interp_config()

    # Save config
    assert save_interp_config(default_config, str(config_file)) is True
    assert config_file.exists()

    # Load config
    loaded = load_interp_config(str(config_file))
    assert loaded is not None
    assert loaded["interpolation"]["max_comparison_methods"] == 4
    assert loaded["interpolation"]["spline_order"] == 3


def test_apply_rbf_interp():
    # Signal with NaN gap
    y = np.array([1.0, 2.0, 3.0, np.nan, np.nan, 6.0, 7.0, 8.0])
    df = pd.DataFrame(y)

    # RBF interpolation on column 0
    y_interp = apply_rbf_interp(df, 0, window_size=3)

    # Verify no NaNs left and all values are finite
    assert not np.isnan(y_interp).any()
    assert np.isfinite(y_interp).all()
    # Check that non-NaN values remain unchanged
    np.testing.assert_equal(y_interp[0], 1.0)
    np.testing.assert_equal(y_interp[7], 8.0)


def test_is_trial_file():
    # True running trials
    assert is_trial_file("s02_d01_t01.csv") is True
    assert is_trial_file("s01_d03_t12.csv") is True
    assert is_trial_file("S99_D99_T99.CSV") is True
    assert is_trial_file("s01_d01_t05_LIMPO.csv") is True
    assert is_trial_file("s01_d01_t05_clean.csv") is True

    # Calibration files
    assert is_trial_file("s02_d01_tara.csv") is False
    assert is_trial_file("s02_d01_peso.csv") is False
    assert is_trial_file("s02_d01_tare.csv") is False
    assert is_trial_file("s02_d01_weight.csv") is False
    assert is_trial_file("s02_d01_10kg.csv") is False
    assert is_trial_file("s02_d01_01kg.csv") is False

    # Sidecars, outputs, Borg / other files
    assert is_trial_file("s01_d01_t05_adjust_intervals.csv") is False
    assert is_trial_file("s01_d01_t05_LIMPO_adjust_intervals.csv") is False
    assert is_trial_file("s01_d01_t05_clean_adjust_intervals.csv") is False
    assert is_trial_file("s01_d01_t05_filter_spectrum_metrics.csv") is False
    assert is_trial_file("s01_d01_t05_processing_steps.csv") is False
    assert is_trial_file("borg_s02_d01.txt") is False
    assert is_trial_file("some_random_file.csv") is False


def test_canonical_trial_filename_normalizes_legacy_limpo_names():
    assert canonical_trial_filename("s01_d01_t05.csv") == "s01_d01_t05.csv"
    assert canonical_trial_filename("s01_d01_t05_LIMPO.csv") == "s01_d01_t05.csv"
    assert canonical_trial_filename("S01_D01_T05_LIMPO.CSV") == "s01_d01_t05.csv"
    assert canonical_trial_filename("s01_d01_t05_clean.csv") == "s01_d01_t05.csv"
    assert canonical_trial_filename("S01_D01_T05_CLEAN.CSV") == "s01_d01_t05.csv"
    assert (
        canonical_trial_filename("s01_d01_t05_adjust_intervals.csv")
        == "s01_d01_t05_adjust_intervals.csv"
    )


def test_deduplicate_trial_files_prefers_standard_name():
    files = ["s01_d01_t02_clean.csv", "s01_d01_t01_clean.csv", "s01_d01_t01.csv"]

    assert deduplicate_trial_files(files) == ["s01_d01_t01.csv", "s01_d01_t02_clean.csv"]


def test_run_adjust_stage_writes_homogeneous_trial_names(tmp_path, monkeypatch):
    trial_adjusted = tmp_path / "s01_d01_t01.csv"
    trial_unchanged = tmp_path / "s01_d01_t02.csv"
    calibration = tmp_path / "s01_d01_tara.csv"
    trial_adjusted.write_text("0,1,2,3,4\n")
    trial_unchanged.write_text("0,5,6,7,8\n")
    calibration.write_text("0,0,0,0,0\n")

    def fake_clean_signal(file_path, parent=None, config=None):
        source = Path(file_path)
        if source.name == "s01_d01_t01.csv":
            adjusted = source.with_name("s01_d01_t01_clean.csv")
            adjusted.write_text("0,9,9,9,9\n")
            return str(adjusted), None, []
        return None, None, []

    monkeypatch.setattr(lct, "clean_signal_with_clicks", fake_clean_signal)
    monkeypatch.setattr(lct.messagebox, "showinfo", lambda *args, **kwargs: None)
    monkeypatch.setattr(lct.messagebox, "showerror", lambda *args, **kwargs: None)

    output_folder = Path(lct.run_adjust_stage(parent=None, initial_dir=str(tmp_path)))

    assert output_folder.parent == tmp_path
    assert output_folder.name == "clean"
    assert (output_folder / "s01_d01_t01.csv").read_text() == "0,9,9,9,9\n"
    assert (output_folder / "s01_d01_t02.csv").read_text() == "0,5,6,7,8\n"
    assert (output_folder / "s01_d01_tara.csv").exists()
    assert not (output_folder / "s01_d01_t01_clean.csv").exists()


def test_is_calibration_file():
    assert is_calibration_file("s02_d01_tare.csv") is True
    assert is_calibration_file("s02_d01_weight.csv") is True
    assert is_calibration_file("s02_d01_tara.csv") is True
    assert is_calibration_file("s02_d01_peso.csv") is True
    assert is_calibration_file("s02_d01_10kg.csv") is True
    assert is_calibration_file("s02_d01_01kg.csv") is True
    assert is_calibration_file("s02_d01_t01.csv") is False
    assert is_calibration_file("s02_d01_t01_adjust_intervals.csv") is False
    assert is_calibration_file("s01_d01_tare_filter_spectrum_metrics.csv") is False
    assert is_calibration_file("borg_s02_d01.txt") is False


def test_get_group_weight_from_borg(tmp_path):
    borg_file = tmp_path / "borg_s01_d03.txt"
    content = (
        "Subject,Day,Trial,Weight,BORG,SBP,DBP,HR,SpO2,HROx,Speed\n"
        "S01,03,T01,61.4,2,138,72,124,96,131,15\n"
        "S01,03,T02*,61.4,2,135,78,112,97,111,15\n"
        "S01,03,T03*,61.4,3,132,83,134,97,120,15\n"
    )
    borg_file.write_text(content)

    assert get_group_weight_from_borg(str(borg_file)) == 61.4


def test_get_group_weight_from_borg_accepts_legacy_portuguese_headers(tmp_path):
    borg_file = tmp_path / "borg_s01_d03.txt"
    content = (
        "Suj,Dia,Tent,Peso,BORG,PAS,PAD,FC,SpO2,FCOx,Vel\nS01,03,T01,61.4,2,138,72,124,96,131,15\n"
    )
    borg_file.write_text(content)

    assert get_group_weight_from_borg(str(borg_file)) == 61.4


def test_get_group_weight_from_info_file(tmp_path):
    info_file = tmp_path / "info_s01_d01.txt"
    info_file.write_text("Subject,Day,Trial,BORG,Speed,Weight\nS01,01,T01,3,15,61.6\n")

    assert get_group_weight_from_borg(str(info_file)) == 61.6


def test_infer_subject_weight_kg_uses_matching_info_file(tmp_path):
    (tmp_path / "s01_d01_t01.csv").write_text("0,0,0,0,0\n", encoding="utf-8")
    (tmp_path / "info_s01_d01.txt").write_text(
        "Subject,Day,Trial,BORG,Speed,Weight\nS01,01,T01,3,15,61.6\n",
        encoding="utf-8",
    )

    assert lct.infer_subject_weight_kg(tmp_path) == 61.6


def test_gui_weight_summary_is_explicit_before_and_after_subject_selection(tmp_path):
    summary_values = []
    body_weight = SimpleNamespace(
        value="",
        set=lambda value: setattr(body_weight, "value", value),
        get=lambda: body_weight.value,
    )
    fake_dialog = SimpleNamespace(
        pipeline_config=lct.get_default_unified_config(),
        input_dir_var=SimpleNamespace(get=lambda: ""),
        body_weight_var=body_weight,
        summary_text_var=SimpleNamespace(set=summary_values.append),
        _write_log=lambda message: None,
    )
    lct.LoadCellTreadmillDialog._update_summary(fake_dialog)
    assert "Not detected" in body_weight.value
    assert "Not detected" in summary_values[-1]

    (tmp_path / "s01_d01_t01.csv").write_text("0,0,0,0,0\n", encoding="utf-8")
    (tmp_path / "info_s01_d01.txt").write_text(
        "Subject,Day,Trial,BORG,Speed,Weight\nS01,01,T01,3,15,61.6\n",
        encoding="utf-8",
    )
    fake_dialog.input_dir_var = SimpleNamespace(get=lambda: str(tmp_path))
    lct.LoadCellTreadmillDialog._sync_subject_weight(fake_dialog, tmp_path)
    lct.LoadCellTreadmillDialog._update_summary(fake_dialog)
    assert body_weight.value == "61.6 (info/Borg)"
    assert "61.6 (info/Borg)" in summary_values[-1]


def test_make_output_dir_reuses_stable_path_and_supports_timestamp_opt_in(tmp_path):
    stable = Path(lct.make_output_dir(tmp_path, "results"))
    (stable / "stale.txt").write_text("old", encoding="utf-8")
    reused = Path(lct.make_output_dir(tmp_path, "results"))

    assert reused == stable
    assert not (reused / "stale.txt").exists()

    timestamped = Path(lct.make_output_dir(tmp_path, "results", timestamp_output=True))
    assert timestamped.name.startswith("results_")
    assert timestamped != stable


def test_discover_calibration_and_borg(tmp_path):
    (tmp_path / "s01_d03_tare.csv").touch()
    (tmp_path / "s01_d03_weight.csv").touch()
    (tmp_path / "s01_d03_05kg.csv").touch()
    (tmp_path / "s01_d03_10kg.csv").touch()
    (tmp_path / "borg_s01_d03.txt").touch()

    subdir = tmp_path / "filtered"
    subdir.mkdir()
    (subdir / "s01_d03_t01.csv").touch()

    tare, weight, plates, borg = discover_calibration_and_borg(str(subdir), "01", "03")

    assert tare == str(tmp_path / "s01_d03_tare.csv")
    assert weight == str(tmp_path / "s01_d03_weight.csv")
    assert len(plates) == 2
    assert str(tmp_path / "s01_d03_05kg.csv") in plates
    assert str(tmp_path / "s01_d03_10kg.csv") in plates
    assert borg == str(tmp_path / "borg_s01_d03.txt")


def test_discover_calibration_accepts_legacy_portuguese_filenames(tmp_path):
    (tmp_path / "s01_d03_tara.csv").touch()
    (tmp_path / "s01_d03_peso.csv").touch()
    (tmp_path / "info_s01_d03.txt").touch()
    subdir = tmp_path / "filtered"
    subdir.mkdir()
    (subdir / "s01_d03_t01.csv").touch()

    tare, weight, plates, borg = discover_calibration_and_borg(str(subdir), "01", "03")

    assert tare == str(tmp_path / "s01_d03_tara.csv")
    assert weight == str(tmp_path / "s01_d03_peso.csv")
    assert plates == []
    assert borg == str(tmp_path / "info_s01_d03.txt")


def test_calibration_center_slice_uses_middle_time_window():
    t = np.arange(10, dtype=float)
    df = pd.DataFrame(
        {
            0: t,
            1: np.r_[100.0, 100.0, np.full(6, 10.0), 100.0, 100.0],
            2: np.r_[200.0, 200.0, np.full(6, 20.0), 200.0, 200.0],
            3: np.r_[300.0, 300.0, np.full(6, 30.0), 300.0, 300.0],
            4: np.r_[400.0, 400.0, np.full(6, 40.0), 400.0, 400.0],
        }
    )

    sliced = calibration_center_slice(df, window_seconds=5.0, fs=1)

    assert sliced[0].min() >= 2.0
    assert sliced[0].max() <= 7.0
    np.testing.assert_allclose(sliced[[1, 2, 3, 4]].mean(axis=0), [10.0, 20.0, 30.0, 40.0])


def test_read_calibration_cells_uses_middle_sample_window_without_valid_time(tmp_path):
    calibration_file = tmp_path / "s01_d01_10kg.csv"
    df = pd.DataFrame(
        {
            0: np.zeros(10),
            1: np.r_[100.0, 100.0, np.full(6, 10.0), 100.0, 100.0],
            2: np.r_[200.0, 200.0, np.full(6, 20.0), 200.0, 200.0],
            3: np.r_[300.0, 300.0, np.full(6, 30.0), 300.0, 300.0],
            4: np.r_[400.0, 400.0, np.full(6, 40.0), 400.0, 400.0],
        }
    )
    df.to_csv(calibration_file, header=False, index=False)

    cells = read_calibration_cells(str(calibration_file), window_seconds=6.0, fs=1)

    assert cells.shape == (6, 4)
    np.testing.assert_allclose(cells.mean(axis=0), [-10.0, -20.0, -30.0, -40.0])


def test_detect_steps_legacy_valley_segments_cut_to_cut_with_internal_peak():
    contact = np.array(
        [
            0.2,
            0.6,
            1.0,
            0.6,
            0.2,
            0.7,
            1.1,
            0.7,
            0.2,
            0.6,
            0.9,
            0.6,
            0.2,
            0.8,
            1.2,
            0.8,
            0.2,
            0.6,
            1.0,
            0.6,
            0.2,
        ]
    )
    grf_total = np.r_[np.zeros(20), contact, np.zeros(20)]

    steps, peaks = detect_steps(grf_total, fs=10, threshold=0.1, mode="legacy_valley")

    assert len(steps) >= 3
    assert len(peaks) >= 3
    first = steps[0]
    assert first["detection_mode"] == "legacy_valley"
    assert first["idx_start"] < first["legacy_peak_index"] < first["idx_end"]
    assert first["sidefoot"] == 0
    assert "side" not in first
    assert "foot" not in first
    assert steps[1]["sidefoot"] == 1
    attrs = strikeattr(first["legacy_signal"], fs=10)
    assert attrs["t_to_peak_s"] > 0
    assert attrs["n_peaks"] > 0


def test_strikeattr_returns_legacy_transient_metrics_for_clear_strike():
    strike = np.array([0.2, 0.8, 1.2, 1.0, 1.5, 1.3, 0.4])

    attrs = strikeattr(strike, fs=100)

    assert attrs["peak_GRF_BW"] == 1.5
    assert attrs["t_to_peak_s"] > 0
    assert attrs["n_peaks"] > 0
    assert np.isfinite(attrs["itransient1_BW"])
    assert np.isfinite(attrs["imp_to_peak_BW_s"])
    assert np.isfinite(attrs["imp_to_trans1_BW_s"])
    assert np.isfinite(attrs["imp_trans1_to_peak_BW_s"])
    assert np.isfinite(attrs["imp_trans2_to_trans1_BW_s"])


def test_load_data_can_skip_processing_filter(tmp_path, monkeypatch):
    running = tmp_path / "s01_d01_t01.csv"
    tare = tmp_path / "s01_d01_tare.csv"
    weight = tmp_path / "s01_d01_weight.csv"
    pd.DataFrame(np.column_stack([np.arange(10), -np.ones((10, 4))])).to_csv(
        running, header=False, index=False
    )
    pd.DataFrame(np.column_stack([np.arange(10), np.zeros((10, 4))])).to_csv(
        tare, header=False, index=False
    )
    pd.DataFrame(np.column_stack([np.arange(10), -2 * np.ones((10, 4))])).to_csv(
        weight, header=False, index=False
    )

    def fail_filter(*args, **kwargs):
        raise AssertionError("processing filter should not be called")

    monkeypatch.setattr(lct, "butterworth_filter", fail_filter)

    grf_bw, grf_total, _, _ = load_data(
        str(running),
        str(tare),
        str(weight),
        weight_kg=80.0,
        apply_processing_filter=False,
    )

    assert grf_bw.shape == (10, 4)
    assert grf_total.shape == (10,)


def test_calculate_cop_system_uses_declared_cell_layout():
    # Cell order: 1 anterior-left, 2 posterior-left, 3 anterior-right, 4 posterior-right.
    grf_bw = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )

    cop_x, cop_y = calculate_cop_system(grf_bw)

    np.testing.assert_allclose(cop_x, [-29.0, -29.0, 29.0, 29.0, 0.0])
    np.testing.assert_allclose(cop_y, [56.5, -56.5, 56.5, -56.5, 0.0])


def test_apply_adjustment_intervals_nan_preserves_length():
    t = np.arange(6, dtype=float)
    cell_data = np.column_stack([np.arange(6, dtype=float) + col for col in range(4)])

    t_adj, cell_data_adj, records = apply_adjustment_intervals(t, cell_data, [(2, 4)], "nan")

    np.testing.assert_array_equal(t_adj, t)
    assert cell_data_adj.shape == cell_data.shape
    assert np.isnan(cell_data_adj[2:4, :]).all()
    np.testing.assert_array_equal(cell_data_adj[:2, :], cell_data[:2, :])
    np.testing.assert_array_equal(cell_data_adj[4:, :], cell_data[4:, :])
    assert records[0]["mode"] == "nan"
    assert records[0]["start_index"] == 2
    assert records[0]["end_index_exclusive"] == 4


def test_apply_adjustment_intervals_selected_cell_only():
    t = np.arange(6, dtype=float)
    cell_data = np.column_stack([np.arange(6, dtype=float) + col * 10 for col in range(4)])

    t_adj, cell_data_adj, records = apply_adjustment_intervals(
        t, cell_data, [{"start": 2, "end": 4, "cells": [2]}], "nan"
    )

    np.testing.assert_array_equal(t_adj, t)
    assert np.isnan(cell_data_adj[2:4, 2]).all()
    np.testing.assert_array_equal(cell_data_adj[:, 0], cell_data[:, 0])
    np.testing.assert_array_equal(cell_data_adj[:, 1], cell_data[:, 1])
    np.testing.assert_array_equal(cell_data_adj[:, 3], cell_data[:, 3])
    assert records[0]["cells_0based"] == [2]
    assert records[0]["cells_1based"] == [3]


def test_apply_adjustment_intervals_remove_shortens_signal():
    t = np.arange(6, dtype=float)
    cell_data = np.column_stack([np.arange(6, dtype=float) + col for col in range(4)])

    t_adj, cell_data_adj, records = apply_adjustment_intervals(t, cell_data, [(2, 4)], "remove")

    np.testing.assert_array_equal(t_adj, np.array([0.0, 1.0, 4.0, 5.0]))
    assert cell_data_adj.shape == (4, 4)
    assert records[0]["mode"] == "remove"
    assert records[0]["samples"] == 2


def test_apply_adjustment_intervals_zero_neutral_and_linear():
    t = np.arange(6, dtype=float)
    cell_data = np.column_stack([np.arange(6, dtype=float) for _ in range(4)])

    _, zeroed, _ = apply_adjustment_intervals(t, cell_data, [(2, 4)], "zero")
    assert np.all(zeroed[2:4, :] == 0.0)

    _, neutral, _ = apply_adjustment_intervals(t, cell_data, [(2, 4)], "neutral")
    # Boundary bridge between samples 1 and 4 gives [2, 3], neutral mean is 2.5.
    np.testing.assert_allclose(neutral[2:4, :], 2.5)

    _, linear, _ = apply_adjustment_intervals(t, cell_data, [(2, 4)], "linear")
    np.testing.assert_allclose(linear[2:4, 0], np.array([2.0, 3.0]))


def test_adjustment_mode_aliases():
    assert normalize_adjustment_mode("nulo") == "nan"
    assert normalize_adjustment_mode("média") == "neutral_mean"
    assert normalize_adjustment_mode("cortar") == "remove"


def test_save_adjustment_metadata_json_toml_csv(tmp_path):
    trial = tmp_path / "s01_d01_t01.csv"
    trial.write_text("0,1,2,3,4\n")
    records = [
        {
            "start_index": 2,
            "end_index_exclusive": 4,
            "end_index_inclusive": 3,
            "start_time_s": 0.002,
            "end_time_s": 0.003,
            "samples": 2,
            "mode": "nan",
        }
    ]

    paths = save_adjustment_metadata(
        str(trial),
        records,
        "adjusted_and_interpolated",
        interpolation_metadata={
            "status": "adjusted_and_interpolated",
            "selected_methods": ["linear", "pchip"],
            "final_method": "pchip",
        },
    )

    assert len(paths) == 3
    for path in paths:
        assert os.path.exists(path)
    assert (tmp_path / "s01_d01_t01_adjust_intervals.json").exists()
    assert (tmp_path / "s01_d01_t01_adjust_intervals.toml").exists()
    assert (tmp_path / "s01_d01_t01_adjust_intervals.csv").exists()
    loaded = load_adjustment_metadata(str(tmp_path / "s01_d01_t01_LIMPO.csv"))
    assert loaded is not None
    assert loaded["interpolation"]["final_method"] == "pchip"


def test_find_and_load_adjustment_metadata_for_legacy_clean_file(tmp_path):
    clean_trial = tmp_path / "s01_d01_t01_LIMPO.csv"
    clean_trial.write_text("0,1,2,3,4\n")
    records = [
        {
            "start_index": 2,
            "end_index_exclusive": 4,
            "end_index_inclusive": 3,
            "start_time_s": 0.002,
            "end_time_s": 0.003,
            "samples": 2,
            "mode": "nan",
            "cells_0based": [2],
            "cells_1based": [3],
            "cell_labels": ["Cell 3"],
        }
    ]
    save_adjustment_metadata(str(tmp_path / "s01_d01_t01.csv"), records, "nan")

    sidecar = find_adjustment_metadata_file(str(clean_trial))
    metadata = load_adjustment_metadata(str(clean_trial))

    assert sidecar == tmp_path / "s01_d01_t01_adjust_intervals.json"
    assert metadata is not None
    assert metadata["intervals"][0]["cells_0based"] == [2]


def test_apply_adjustment_metadata_as_nan_selected_cells_only():
    df = pd.DataFrame(np.arange(24, dtype=float).reshape(6, 4))
    metadata = {
        "intervals": [
            {
                "start_index": 2,
                "end_index_exclusive": 4,
                "cells_0based": [2],
            }
        ]
    }

    applied = apply_adjustment_metadata_as_nan(df, metadata)

    assert applied == [(2, 4, [2])]
    assert np.isnan(df.loc[2:3, 2]).all()
    assert not np.isnan(df.loc[2:3, 0]).any()
    assert not np.isnan(df.loc[2:3, 1]).any()
    assert not np.isnan(df.loc[2:3, 3]).any()


def test_adjustment_metadata_to_interval_specs_uses_shared_shape():
    metadata = {
        "intervals": [
            {
                "start_index": 2,
                "end_index_exclusive": 4,
                "cells_0based": [2],
            }
        ]
    }

    specs = adjustment_metadata_to_interval_specs(metadata)

    assert specs == [{"start": 2, "end": 4, "cells": [2]}]


def test_preprocess_file_interp_skips_already_interpolated_sidecar(tmp_path):
    clean_trial = tmp_path / "s01_d01_t01_LIMPO.csv"
    data = np.column_stack((np.arange(6, dtype=float), np.ones((6, 4))))
    np.savetxt(clean_trial, data, delimiter=",")
    records = [
        {
            "start_index": 2,
            "end_index_exclusive": 4,
            "end_index_inclusive": 3,
            "start_time_s": 2.0,
            "end_time_s": 3.0,
            "samples": 2,
            "mode": "adjusted_and_interpolated",
            "cells_0based": [2],
            "cells_1based": [3],
            "cell_labels": ["Cell 3"],
        }
    ]
    save_adjustment_metadata(
        str(tmp_path / "s01_d01_t01.csv"),
        records,
        "adjusted_and_interpolated",
        interpolation_metadata={"status": "adjusted_and_interpolated", "final_method": "pchip"},
    )

    saved, _, _, did_interpolate = preprocess_file_interp(
        str(clean_trial), get_default_interp_config(), root=None
    )

    assert did_interpolate is False
    np.testing.assert_allclose(saved, data)


def test_lowpass_filter_preserves_constant_signal_edges():
    signal = np.full(1000, 42.0)

    filtered = apply_filter(
        signal,
        filter_type="lowpass",
        fs=1000,
        median_window=5,
        edge_mode="nearest",
        lowpass_cutoff=40.0,
        order=4,
    )

    np.testing.assert_allclose(filtered, signal, atol=1e-8)


def test_old_filter_toml_defaults_to_lowpass(tmp_path):
    config_path = tmp_path / "old_filter.toml"
    config_path.write_text(
        "[filters]\n"
        "median_window = 13\n"
        "bandpass_lowcut = 0.5\n"
        "bandpass_highcut = 40.0\n"
        "filter_order = 4\n",
        encoding="utf-8",
    )

    config = load_filter_config(str(config_path))

    assert config is not None
    filters = config["filters"]
    assert filters["filter_type"] == "lowpass"
    assert filters["median_window"] == 13
    assert filters["lowpass_cutoff"] == 40.0
    assert filters["bandpass_lowcut"] == 0.5
    assert filters["bandpass_highcut"] == 40.0
    assert filters["filter_order"] == 4
    assert filters["edge_mode"] == "nearest"


def test_normalize_analysis_window_points_enter_after_start_uses_signal_end():
    assert normalize_analysis_window_points([(123.4, 1.0)], 1000) == (123, 1000)


def test_normalize_analysis_window_points_two_clicks_use_interval():
    assert normalize_analysis_window_points([(100.2, 1.0), (900.6, 1.0)], 1000) == (100, 901)


def test_normalize_analysis_window_points_invalid_selection_returns_none():
    assert normalize_analysis_window_points([], 1000) is None
    assert normalize_analysis_window_points([(1, 1.0), (2, 1.0), (3, 1.0)], 1000) is None
    assert normalize_analysis_window_points([(800, 1.0), (200, 1.0)], 1000) is None


def test_plot_trial_figures_writes_overview_and_full_cop(tmp_path):
    grf_total = np.linspace(0.0, 1.0, 20)
    cop_x = np.linspace(-0.2, 0.2, 20)
    cop_y = np.sin(np.linspace(0.0, np.pi, 20)) * 0.1
    steps = [{"idx_start": 2, "idx_end": 8, "foot": "R"}]
    peaks = np.array([5])

    plot_trial_figures(grf_total, cop_x, cop_y, steps, peaks, "s01_d01_t01.csv", tmp_path)

    assert (tmp_path / "s01_d01_t01_processing_overview.png").exists()
    assert (tmp_path / "s01_d01_t01_processing_strike_attributes.png").exists()
    assert (tmp_path / "s01_d01_t01_processing_stride_map.png").exists()
    assert (tmp_path / "s01_d01_t01_processing_cop_trajectory.png").exists()
    report = tmp_path / "s01_d01_t01_processing_cop_report_interactive.html"
    assert report.exists()
    report_text = report.read_text(encoding="utf-8")
    assert "COP X - Mediolateral (cm)" in report_text
    assert "COP Y - Anteroposterior (cm)" in report_text
    assert "Total GRF First Derivative" in report_text
    assert "COP Contact-Load Location on 58 x 113 cm Treadmill Deck" in report_text
    assert "not belt displacement and not stride length" in report_text
    assert "Cell 1" in report_text
    assert not list(tmp_path.glob("strike_*.png"))


def test_analyze_spectrum_filt_uses_filter_specific_output_names(tmp_path):
    t = np.linspace(0.0, 1.0, 128, endpoint=False)
    cells = np.column_stack([np.sin(2 * np.pi * (i + 1) * t) for i in range(4)])

    analyze_spectrum_filt(cells, t, "s01_d01_t01.csv", tmp_path, fs=128)

    assert (tmp_path / "s01_d01_t01_filter_Cell_1_spectrum.png").exists()
    assert (tmp_path / "s01_d01_t01_filter_sum_spectrum.png").exists()
    assert (tmp_path / "s01_d01_t01_filter_spectrum_metrics.csv").exists()
    assert not (tmp_path / "s01_d01_t01_metrics.csv").exists()


def test_run_process_stage_skips_excluded_trials(tmp_path, monkeypatch):
    trial = tmp_path / "s01_d01_t01.csv"
    tare = tmp_path / "s01_d01_tare.csv"
    weight = tmp_path / "s01_d01_weight.csv"

    # Write dummy CSV data
    pd.DataFrame(np.zeros((10, 5))).to_csv(trial, header=False, index=False)
    pd.DataFrame(np.zeros((10, 5))).to_csv(tare, header=False, index=False)
    pd.DataFrame(np.ones((10, 5))).to_csv(weight, header=False, index=False)

    # Save excluded metadata for this trial
    records = [
        {"start_index": 0, "end_index_exclusive": 10, "cells_0based": [0], "mode": "excluded"}
    ]
    save_adjustment_metadata(
        str(trial),
        records,
        "excluded",
        interpolation_metadata={"status": "excluded", "processed": False},
    )

    class DummyDialog:
        def __init__(self, parent):
            self.result = {
                "processing": {
                    "participant_weight_kg": 70.0,
                    "use_advanced_calibration": False,
                    "filter_cutoff_hz": 50,
                    "apply_processing_filter": False,
                    "detection_threshold_bw": 0.1,
                    "generate_figures": False,
                }
            }

    monkeypatch.setattr(lct, "ProcessConfigDialog", DummyDialog)
    monkeypatch.setattr(lct.messagebox, "showinfo", lambda *args, **kwargs: None)
    monkeypatch.setattr(lct.messagebox, "showerror", lambda *args, **kwargs: None)

    out_dir = lct.run_process_stage(parent=None, initial_dir=str(tmp_path))

    assert out_dir is not None
    steps_csv = Path(out_dir) / "s01_d01_t01_processing_steps.csv"
    assert not steps_csv.exists()

    metrics_csv = Path(out_dir) / "s01_d01_processing_metrics.csv"
    assert not metrics_csv.exists()


def test_plot_trial_figures_uses_agg_backend_not_qt(tmp_path):
    """Regression: savefig paths must not load QtAgg/PySide6 (hangs/aborts in pytest)."""
    assert matplotlib.get_backend().lower() == "agg"
    plot_trial_figures(
        np.linspace(0.0, 1.0, 20),
        np.linspace(-0.2, 0.2, 20),
        np.sin(np.linspace(0.0, np.pi, 20)) * 0.1,
        [{"idx_start": 2, "idx_end": 8, "foot": "R"}],
        np.array([5]),
        "s01_d01_t01.csv",
        tmp_path,
        generate_interactive_report=False,
    )
    assert matplotlib.get_backend().lower() == "agg"
    assert (tmp_path / "s01_d01_t01_processing_overview.png").exists()


# =============================================================================
# 5-STAGE LOGICAL PIPELINE & REFACTOR TESTS
# =============================================================================


def test_apply_zero_offset_subtracts_mean_baseline():
    """Stage 1: Test zero offset subtraction using baseline data."""
    raw = np.array([[10.0, 20.0, 30.0, 40.0], [12.0, 22.0, 32.0, 42.0]])
    baseline = np.array([[2.0, 4.0, 6.0, 8.0], [2.0, 4.0, 6.0, 8.0]])
    zeroed = lct.apply_zero_offset(raw, baseline)
    expected = np.array([[8.0, 16.0, 24.0, 32.0], [10.0, 18.0, 26.0, 34.0]])
    np.testing.assert_allclose(zeroed, expected)


def test_apply_calibration_matrix_linear_scaling():
    """Stage 2: Test calibration matrix linear scaling into Body Weight (BW)."""
    zeroed = np.array([[100.0, 100.0, 100.0, 100.0]])
    # Calibration slope m=0.5, intercept b=0.0, weight_kg=100.0
    # Scaled sum = (0.5 * 400.0 + 0) / 100.0 = 2.0 BW
    grf_bw, grf_total = lct.apply_calibration_matrix(
        zeroed, weight_kg=100.0, calib_slope=0.5, calib_intercept=0.0
    )
    np.testing.assert_allclose(grf_total, [2.0])
    np.testing.assert_allclose(grf_bw, [[0.5, 0.5, 0.5, 0.5]])


def test_apply_coordinate_transformation_geometry_and_keys():
    """Stage 3: Test coordinate transformation yielding medial_lateral, anterior_posterior, vertical."""
    # Deck geometry: X: +/-29 cm, Y: +/-56.5 cm
    # Cell 1: AL (-29, +56.5), Cell 2: PL (-29, -56.5)
    # Cell 3: AR (+29, +56.5), Cell 4: PR (+29, -56.5)
    grf_bw = np.array([[0.5, 0.5, 0.5, 0.5]])  # perfectly symmetric load
    coords = lct.apply_coordinate_transformation(grf_bw)
    assert "medial_lateral" in coords
    assert "anterior_posterior" in coords
    assert "vertical" in coords
    # Symmetric load gives COP at origin (0, 0)
    np.testing.assert_allclose(coords["medial_lateral"], [0.0])
    np.testing.assert_allclose(coords["anterior_posterior"], [0.0])
    np.testing.assert_allclose(coords["vertical"], [2.0])


def test_apply_signal_filtering_zero_phase():
    """Stage 4: Test signal filtering on channels."""
    t = np.linspace(0, 1, 1000)
    sig = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 100 * t)
    data = np.tile(sig[:, None], (1, 4))
    filt_data = lct.apply_signal_filtering(data, fs=1000, cutoff_hz=20.0, filter_order=4)
    assert filt_data.shape == data.shape
    # High frequency component (100 Hz) should be substantially attenuated
    assert np.std(filt_data[:, 0]) < np.std(data[:, 0])


def test_detect_events_strict_sidefoot_0_right_1_left():
    """Stage 5: Event detection uses a single sidefoot code (0=Right, 1=Left)."""
    contact = np.array(
        [
            0.2,
            0.6,
            1.0,
            0.6,
            0.2,
            0.7,
            1.1,
            0.7,
            0.2,
            0.6,
            0.9,
            0.6,
            0.2,
            0.8,
            1.2,
            0.8,
            0.2,
            0.6,
            1.0,
            0.6,
            0.2,
        ]
    )
    grf_total = np.r_[np.zeros(20), contact, np.zeros(20)]

    steps, peaks = lct.detect_events(grf_total, start=0, fs=10, threshold=0.1, mode="legacy_valley")
    assert len(steps) >= 2
    # Canonical encoding: first step Right (0), second step Left (1).
    assert steps[0]["sidefoot"] == 0
    assert steps[1]["sidefoot"] == 1
    assert "side" not in steps[0]
    assert "foot" not in steps[0]


def test_calculate_kinetic_metrics_strike_english_spatial_keys():
    """Test that strike metrics return both standard English and legacy COP keys."""
    grf_total = np.ones(100) * 1.5
    cop_x = np.linspace(-5.0, 5.0, 100)
    cop_y = np.linspace(10.0, 30.0, 100)
    metrics = lct.calculate_kinetic_metrics_strike(grf_total, 10, 90, cop_x, cop_y, fs=1000)

    # Standard English biomechanical spatial keys
    assert "cop_medial_lateral_mean" in metrics
    assert "cop_anterior_posterior_mean" in metrics
    assert "cop_medial_lateral_range" in metrics
    assert "cop_anterior_posterior_range" in metrics
    assert "cop_anterior_posterior_initial" in metrics
    assert "cop_anterior_posterior_final" in metrics

    # Legacy keys preserved
    assert "cop_x_mean" in metrics
    assert "cop_y_mean" in metrics
    assert metrics["cop_medial_lateral_mean"] == metrics["cop_x_mean"]
    assert metrics["cop_anterior_posterior_mean"] == metrics["cop_y_mean"]


def test_load_unified_config_processing_fixture(tmp_path):
    """Test loading unified TOML configuration from file."""
    toml_content = """
[paths]
input_dir = "data"
output_dir = "results"

[filters]
filter_type = "lowpass"
lowpass_cutoff = 50.0

[processing]
participant_weight_kg = 61.6
filter_cutoff_hz = 50.0
detection_threshold_bw = 0.1
"""
    cfg_file = tmp_path / "processing_configuration_used.toml"
    cfg_file.write_text(toml_content, encoding="utf-8")
    config = lct.load_unified_config(cfg_file)
    assert "processing" in config
    assert "filters" in config
    assert "interpolation" in config
    proc = config["processing"]
    assert proc["participant_weight_kg"] == 61.6
    assert proc["filter_cutoff_hz"] == 50.0
    assert proc["detection_threshold_bw"] == 0.1


def test_main_cli_step_process_headless_with_toml(tmp_path):
    """CLI honors --output-dir and writes complete history to both roots."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "custom-output"
    input_dir.mkdir()
    trial = input_dir / "s01_d01_t01.csv"
    tare = input_dir / "s01_d01_tare.csv"
    weight = input_dir / "s01_d01_weight.csv"
    toml_cfg = tmp_path / "config.toml"

    # Create dummy data
    t = np.linspace(0, 2, 2000)
    cells = np.zeros((2000, 4))
    # Add a contact wave
    cells[500:1500, :] = 0.5 * np.sin(np.pi * (t[500:1500] - 0.5))[:, None]
    df_trial = pd.DataFrame(np.column_stack((t, cells)))
    df_trial.to_csv(trial, header=False, index=False)

    df_tare = pd.DataFrame(np.zeros((100, 5)))
    df_tare.to_csv(tare, header=False, index=False)

    df_weight = pd.DataFrame(np.ones((100, 5)) * 0.25)
    df_weight.to_csv(weight, header=False, index=False)

    toml_cfg.write_text("""
[processing]
participant_weight_kg = 75.0
filter_cutoff_hz = 30.0
detection_threshold_bw = 0.1
generate_figures = false
generate_interactive_report = false
""")

    ret = lct.main(
        [
            "--config",
            str(toml_cfg),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--step",
            "process",
        ]
    )
    assert ret == 0

    res_dir = output_dir / "results"
    assert res_dir.is_dir()
    assert not list(output_dir.glob("results_*"))
    assert (res_dir / "processing_configuration_used.toml").exists()
    assert (res_dir / "s01_d01_t01_processing_steps.csv").exists()
    steps_df = pd.read_csv(res_dir / "s01_d01_t01_processing_steps.csv")
    assert "sidefoot" in steps_df.columns
    assert "side" not in steps_df.columns
    assert "foot" not in steps_df.columns
    if len(steps_df) > 0:
        assert set(steps_df["sidefoot"].dropna().astype(int).unique()).issubset({0, 1})
    assert not list(input_dir.glob("results_*"))

    input_history = input_dir / lct.RUN_HISTORY_FILENAME
    output_history = output_dir / lct.RUN_HISTORY_FILENAME
    assert input_history.is_file()
    assert output_history.is_file()
    assert input_history.read_bytes() == output_history.read_bytes()
    assert list(input_dir.glob("treadmill_lc_run_history_*.toml")) == []
    assert list(output_dir.glob("treadmill_lc_run_history_*.toml")) == []

    history = lct.load_unified_config(input_history)
    assert history["paths"]["input_dir"] == str(input_dir.resolve())
    assert history["paths"]["output_dir"] == str(output_dir.resolve())
    assert history["execution"]["step"] == "process"
    assert history["execution"]["vaila_version"] == lct.VERSION
    assert history["processing"]["participant_weight_kg"] == 75.0
    assert history["filters"] == lct.get_default_filter_config()["filters"]
    assert history["interpolation"] == lct.get_default_interp_config()["interpolation"]


def test_main_cli_defaults_to_input_output_and_history_reruns(tmp_path, monkeypatch, capsys):
    """Without --output-dir, CLI creates INPUT/output and history replays its step."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    calls = []

    def fake_process_stage(parent, initial_dir, config, output_dir, headless):
        calls.append(
            {
                "parent": parent,
                "initial_dir": initial_dir,
                "config": config,
                "output_dir": Path(output_dir),
                "headless": headless,
            }
        )
        result = Path(output_dir) / f"results_test_{len(calls)}"
        result.mkdir()
        return str(result)

    def fail_if_tk_is_created(*args, **kwargs):
        raise AssertionError("Headless CLI must not create a Tk root")

    monkeypatch.setattr(lct, "run_process_stage", fake_process_stage)
    monkeypatch.setattr(lct.tk, "Tk", fail_if_tk_is_created)

    assert lct.main(["--input-dir", str(input_dir), "--step", "process"]) == 0

    output_dir = input_dir / "output"
    assert output_dir.is_dir()
    assert calls[0]["parent"] is None
    assert calls[0]["initial_dir"] == str(input_dir.resolve())
    assert calls[0]["output_dir"] == output_dir.resolve()
    assert calls[0]["headless"] is True

    first_input_history = input_dir / lct.RUN_HISTORY_FILENAME
    first_output_history = output_dir / lct.RUN_HISTORY_FILENAME
    assert first_input_history.is_file()
    assert first_output_history.is_file()
    assert first_input_history.read_bytes() == first_output_history.read_bytes()

    # The generated history contains paths and step, so --config alone replays it.
    assert lct.main(["--config", str(first_input_history)]) == 0
    assert len(calls) == 2
    assert calls[1]["output_dir"] == output_dir.resolve()
    assert list(input_dir.glob(lct.RUN_HISTORY_FILENAME)) == [first_input_history]
    assert list(output_dir.glob(lct.RUN_HISTORY_FILENAME)) == [first_output_history]
    assert list(input_dir.glob("treadmill_lc_run_history_*.toml")) == []
    assert list(output_dir.glob("treadmill_lc_run_history_*.toml")) == []

    terminal_output = capsys.readouterr().out
    assert "Pipeline Step" in terminal_output
    assert "Reproducible TOML History" in terminal_output
    assert "Processing Complete" in terminal_output


def test_gui_run_prints_highlighted_parseable_cli_and_updates_history(tmp_path, capsys):
    """GUI Run mirrors sam3sapiens2: highlighted CLI plus complete mutable history."""
    input_dir = tmp_path / "input data"
    output_dir = tmp_path / "output data"
    input_dir.mkdir()

    config, input_path, output_path, history_paths, cli = lct.prepare_gui_cli_run(
        lct.get_default_unified_config(), input_dir, output_dir, "process"
    )

    assert cli[:3] == ["uv", "run", "vaila/treadmill_lc.py"]
    parsed = lct._parse_args(cli[3:])
    assert Path(parsed.config) == history_paths[0]
    assert history_paths[0].name == lct.RUN_HISTORY_FILENAME
    assert history_paths[1].name == lct.RUN_HISTORY_FILENAME
    assert Path(parsed.input_dir) == input_path
    assert Path(parsed.output_dir) == output_path
    assert parsed.step == "process"

    lct.record_adjustment_in_config(
        config,
        input_path / "s01_d01_t01.csv",
        [{"start_index": 2, "end_index_exclusive": 4, "cells_0based": [0]}],
        "nan",
        {
            "status": "adjusted_and_interpolated",
            "final_method": "linear",
            "spline_order": 3,
            "rbf_window_size": 200,
        },
    )
    lct.record_analysis_window_in_config(config, "s01_d01_t01.csv", 10, 90, 100)
    lct.finish_gui_cli_run(config, history_paths, cli, succeeded=True)

    assert history_paths[0].read_bytes() == history_paths[1].read_bytes()
    replay = lct.load_unified_config(history_paths[0])
    assert replay["execution"]["step"] == "process"
    assert replay["adjustments"]["s01_d01_t01"]["interpolation"]["final_method"] == "linear"
    assert replay["analysis_windows"]["s01_d01_t01"]["start_index"] == 10

    terminal_output = capsys.readouterr().out
    assert ">> vaila/treadmill_lc: Equivalent CLI" in terminal_output
    assert "uv run vaila/treadmill_lc.py" in terminal_output
    assert f"'{input_path}'" in terminal_output
    assert "Equivalent CLI for the completed GUI run" in terminal_output
    assert "s01_d01_t01 Analysis Window" in terminal_output


def test_gui_stage_button_routes_through_cli_mirror_lifecycle(tmp_path, monkeypatch):
    """Regression: the GUI button itself must invoke prepare/finish mirror hooks."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    history = input_dir / "history.toml"
    events = []

    def fake_prepare(config, selected_input, selected_output, stage):
        events.append(("prepare", stage, selected_input, selected_output))
        return config, input_dir, output_dir, [history], ["uv", "run", "fake"]

    def fake_filter_stage(**kwargs):
        events.append(("run", kwargs))
        return str(output_dir / "filtered_test")

    def fake_finish(config, paths, cli, *, succeeded):
        events.append(("finish", succeeded, paths, cli))

    monkeypatch.setattr(lct, "prepare_gui_cli_run", fake_prepare)
    monkeypatch.setattr(lct, "run_filter_stage", fake_filter_stage)
    monkeypatch.setattr(lct, "finish_gui_cli_run", fake_finish)

    fake_dialog = SimpleNamespace(
        pipeline_config=lct.get_default_unified_config(),
        input_dir_var=SimpleNamespace(set=lambda value: events.append(("input", value))),
        output_dir_var=SimpleNamespace(set=lambda value: events.append(("output", value))),
        _get_target_dirs=lambda: (str(input_dir), str(output_dir)),
        _write_log=lambda message: events.append(("log", message)),
    )

    lct.LoadCellTreadmillDialog._execute_stage(fake_dialog, "filter")

    assert events[0][0] == "prepare"
    run_event = next(event for event in events if event[0] == "run")
    assert run_event[1]["parent"] is None
    assert run_event[1]["initial_dir"] == str(input_dir)
    assert run_event[1]["output_dir"] == output_dir
    assert run_event[1]["headless"] is True
    assert events[-1] == ("finish", True, [history], ["uv", "run", "fake"])


def test_gui_full_pipeline_runs_continuously_without_stage_dialogs(tmp_path, monkeypatch):
    """GUI Run must consume TOML choices through the same non-interactive path as CLI."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    history = input_dir / "history.toml"
    config = lct.get_default_unified_config()
    lct.record_analysis_window_in_config(config, "s01_d01_t01.csv", 526, 60000, 60000)
    events = []

    def fake_prepare(selected_config, selected_input, selected_output, stage):
        events.append(("prepare", stage))
        return selected_config, input_dir, output_dir, [history], ["uv", "run", "fake"]

    def stage_result(name, result):
        def fake_stage(**kwargs):
            events.append((name, kwargs))
            return str(result)

        return fake_stage

    def fake_finish(selected_config, paths, cli, *, succeeded):
        events.append(("finish", succeeded, paths, cli))

    def fail_on_success_popup(*args, **kwargs):
        raise AssertionError("Continuous GUI Run must not open success/confirmation popups")

    monkeypatch.setattr(lct, "prepare_gui_cli_run", fake_prepare)
    monkeypatch.setattr(lct, "run_filter_stage", stage_result("filter", output_dir / "filtered"))
    monkeypatch.setattr(lct, "run_adjust_stage", stage_result("adjust", output_dir / "clean"))
    monkeypatch.setattr(
        lct, "run_interpolate_stage", stage_result("interpolate", output_dir / "adjusted")
    )
    monkeypatch.setattr(lct, "run_process_stage", stage_result("process", output_dir / "results"))
    monkeypatch.setattr(lct, "finish_gui_cli_run", fake_finish)
    monkeypatch.setattr(lct.messagebox, "showinfo", fail_on_success_popup)

    fake_dialog = SimpleNamespace(
        pipeline_config=config,
        input_dir_var=SimpleNamespace(set=lambda value: events.append(("input", value))),
        output_dir_var=SimpleNamespace(set=lambda value: events.append(("output", value))),
        _get_target_dirs=lambda: (str(input_dir), str(output_dir)),
        _write_log=lambda message: events.append(("log", message)),
    )

    lct.LoadCellTreadmillDialog._run_full_pipeline(fake_dialog)

    stage_events = [
        event for event in events if event[0] in {"filter", "adjust", "interpolate", "process"}
    ]
    assert [event[0] for event in stage_events] == ["filter", "adjust", "interpolate", "process"]
    for _, kwargs in stage_events:
        assert kwargs["parent"] is None
        assert kwargs["headless"] is True
        assert kwargs["config"]["analysis_windows"]["s01_d01_t01"]["start_index"] == 526
    assert events[-1] == ("finish", True, [history], ["uv", "run", "fake"])


def test_gui_process_button_reuses_toml_window_without_foot_strike_dialog(tmp_path, monkeypatch):
    """Regression: a loaded analysis window must bypass the foot-strike picker."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    sample_count = 800
    time = np.arange(sample_count, dtype=float) / lct.FS
    trial_cells = np.zeros((sample_count, 4))
    trial_cells[550:700] = 0.5
    pd.DataFrame(np.column_stack((time, trial_cells))).to_csv(
        input_dir / "s01_d01_t01.csv", header=False, index=False
    )
    pd.DataFrame(np.zeros((100, 5))).to_csv(
        input_dir / "s01_d01_tare.csv", header=False, index=False
    )
    pd.DataFrame(np.ones((100, 5)) * 0.25).to_csv(
        input_dir / "s01_d01_weight.csv", header=False, index=False
    )

    config = lct.get_default_unified_config()
    config["processing"].update({"generate_figures": False, "generate_interactive_report": False})
    lct.record_analysis_window_in_config(config, "s01_d01_t01.csv", 526, sample_count, sample_count)
    selector_calls = []
    popup_calls = []

    def fake_prepare(selected_config, selected_input, selected_output, stage):
        return (
            selected_config,
            input_dir,
            output_dir,
            [input_dir / "history.toml"],
            ["uv", "run", "fake"],
        )

    def track_selector(*args, **kwargs):
        selector_calls.append((args, kwargs))
        return 0, sample_count

    def track_popup(*args, **kwargs):
        popup_calls.append((args, kwargs))

    monkeypatch.setattr(lct, "prepare_gui_cli_run", fake_prepare)
    monkeypatch.setattr(lct, "finish_gui_cli_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(lct, "select_analysis_window", track_selector)
    monkeypatch.setattr(lct.messagebox, "showinfo", track_popup)

    fake_dialog = SimpleNamespace(
        pipeline_config=config,
        input_dir_var=SimpleNamespace(set=lambda value: None),
        output_dir_var=SimpleNamespace(set=lambda value: None),
        _get_target_dirs=lambda: (str(input_dir), str(output_dir)),
        _write_log=lambda message: None,
    )

    lct.LoadCellTreadmillDialog._execute_stage(fake_dialog, "process")

    assert selector_calls == []
    assert popup_calls == []
    result_dirs = [output_dir / "results"]
    assert result_dirs[0].is_dir()
    saved_config = lct.load_unified_config(result_dirs[0] / "processing_configuration_used.toml")
    assert saved_config["analysis_windows"]["s01_d01_t01"]["start_index"] == 526
    assert saved_config["analysis_windows"]["s01_d01_t01"]["end_index_exclusive"] == sample_count


def test_headless_interpolation_compatibility_never_opens_gui(tmp_path, monkeypatch):
    """Incomplete legacy sidecars must not break a continuous GUI/CLI run."""
    trial = tmp_path / "s01_d01_t01.csv"
    time = np.arange(8, dtype=float)
    cells = np.column_stack((time, time + 10, time + 20, time + 30))
    np.savetxt(trial, np.column_stack((time, cells)), delimiter=",")
    lct.save_adjustment_metadata(
        trial,
        [{"start_index": 2, "end_index_exclusive": 4, "cells_0based": [0]}],
        "nan",
        interpolation_metadata={"status": "pending"},
    )

    def fail_if_gui_is_opened(*args, **kwargs):
        raise AssertionError("Headless interpolation compatibility must not open Tk/Matplotlib GUI")

    monkeypatch.setattr(lct, "_ensure_interactive_backend", fail_if_gui_is_opened)
    monkeypatch.setattr(lct.tk, "Tk", fail_if_gui_is_opened)
    monkeypatch.setattr(lct.messagebox, "showinfo", fail_if_gui_is_opened)

    saved, _, _, did_interpolate = lct.preprocess_file_interp(
        trial,
        lct.get_default_unified_config(),
        headless=True,
    )

    np.testing.assert_allclose(saved, np.column_stack((time, cells)))
    assert did_interpolate is False


def test_printed_gui_cli_replays_adjustment_without_gui(tmp_path, monkeypatch):
    """The printed GUI history must reproduce adjustment without Tk dialogs."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    trial = input_dir / "s01_d01_t01.csv"
    t = np.arange(6, dtype=float)
    cells = np.column_stack([t, t + 10, t + 20, t + 30])
    np.savetxt(trial, np.column_stack((t, cells)), delimiter=",")

    config = lct.get_default_unified_config()
    lct.record_adjustment_in_config(
        config,
        trial,
        [
            {
                "start_index": 2,
                "end_index_exclusive": 4,
                "cells_0based": [0],
                "mode": "adjusted_and_interpolated",
            }
        ],
        "nan",
        {
            "status": "adjusted_and_interpolated",
            "selected_methods": ["linear"],
            "final_method": "linear",
            "spline_order": 3,
            "rbf_window_size": 200,
        },
    )

    config, _, _, history_paths, cli = lct.prepare_gui_cli_run(
        config, input_dir, output_dir, "adjust"
    )
    lct.finish_gui_cli_run(config, history_paths, cli, succeeded=True)

    def fail_if_tk_is_created(*args, **kwargs):
        raise AssertionError("The printed GUI CLI must replay without Tk")

    monkeypatch.setattr(lct.tk, "Tk", fail_if_tk_is_created)
    assert lct.main(cli[3:]) == 0

    clean_dir = output_dir / "clean"
    assert clean_dir.is_dir()
    replayed = pd.read_csv(clean_dir / trial.name, header=None).to_numpy()

    np.testing.assert_allclose(replayed, np.column_stack((t, cells)))
    assert (clean_dir / "s01_d01_t01_adjust_intervals.toml").exists()


def test_recorded_analysis_window_is_used_for_headless_replay():
    config = lct.get_default_unified_config()
    lct.record_analysis_window_in_config(config, "s01_d01_t01.csv", 25, 80, 100)

    assert lct.get_recorded_analysis_window(config, "s01_d01_t01.csv", 100) == (25, 80)
    # Stored limits remain safe if the replayed signal is shorter.
    assert lct.get_recorded_analysis_window(config, "s01_d01_t01.csv", 60) == (25, 60)


def test_detect_signal_start_index_finds_first_peak_above_proportional_threshold():
    signal = np.zeros(260)
    signal[1] = 6.0  # Local oscillation within 100 ms of the first real impact.
    signal[10] = 4.9
    signal[50] = 10.0
    signal[190] = 8.0

    assert lct.detect_signal_start_index(signal) == 50
    assert lct.detect_signal_start_index(np.ones(20)) == 0
    assert lct.detect_signal_start_index(np.array([np.nan, np.nan])) == 0


def test_analysis_window_priority_is_cli_then_toml_then_automatic():
    signal = np.zeros(100)
    signal[40] = 10.0
    config = lct.get_default_unified_config()
    lct.record_analysis_window_in_config(config, "s01_d01_t01.csv", 20, 90, len(signal))

    assert lct.resolve_analysis_window(config, "s01_d01_t01.csv", signal) == (
        20,
        90,
        "toml",
        "toml",
    )

    config["execution"]["start_index"] = 30
    assert lct.resolve_analysis_window(config, "s01_d01_t01.csv", signal) == (
        30,
        90,
        "cli",
        "toml",
    )

    config["execution"].update({"force_auto": True, "start_index": None})
    assert lct.resolve_analysis_window(config, "s01_d01_t01.csv", signal) == (
        40,
        100,
        "automatic",
        "automatic",
    )


def test_cli_parser_exposes_automatic_processing_arguments():
    defaults = lct._parse_args([])
    assert defaults.toml_path == lct.DEFAULT_TOML_PATH

    parsed = lct._parse_args(
        [
            "--weight",
            "82.5",
            "--start-index",
            "526",
            "--end-index",
            "60000",
            "--force-auto",
            "--toml-path",
            "custom.toml",
        ]
    )
    assert parsed.weight == 82.5
    assert parsed.start_index == 526
    assert parsed.end_index == 60000
    assert parsed.force_auto is True
    assert parsed.toml_path == "custom.toml"

    short = lct._parse_args(
        [
            "-i",
            "input",
            "-o",
            "output",
            "-t",
            "config.toml",
            "-s",
            "process",
            "-w",
            "61.6",
            "-b",
            "526",
            "-e",
            "60000",
            "-a",
            "-T",
        ]
    )
    assert short.input_dir == "input"
    assert short.output_dir == "output"
    assert short.toml_path == "config.toml"
    assert short.step == "process"
    assert short.weight == 61.6
    assert short.start_index == 526
    assert short.end_index == 60000
    assert short.force_auto is True
    assert short.timestamp_output is True


def test_cli_overrides_toml_weight_and_window_fields(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    config_path = tmp_path / "processing.toml"
    config_path.write_text(
        """
weight = 72.0

[analysis_windows.s01_d01_t01]
start_index = 20
end_index_exclusive = 90
source_samples = 100
""",
        encoding="utf-8",
    )
    toml_config = lct.load_unified_config(config_path)
    assert toml_config["processing"]["participant_weight_kg"] == 72.0
    assert toml_config["processing"]["participant_weight_source"] == "toml"
    calls = []

    def fake_process_stage(parent, initial_dir, config, output_dir, headless):
        calls.append(config)
        result = Path(output_dir) / "results_test"
        result.mkdir()
        return str(result)

    monkeypatch.setattr(lct, "run_process_stage", fake_process_stage)

    assert (
        lct.main(
            [
                "--input-dir",
                str(input_dir),
                "--toml-path",
                str(config_path),
                "--step",
                "process",
                "--weight",
                "82.5",
                "--start-index",
                "30",
            ]
        )
        == 0
    )

    effective = calls[0]
    assert effective["processing"]["participant_weight_kg"] == 82.5
    assert effective["processing"]["participant_weight_source"] == "cli"
    assert lct.resolve_analysis_window(
        effective, "s01_d01_t01.csv", np.r_[np.zeros(40), 10.0, np.zeros(59)]
    ) == (30, 90, "cli", "toml")


def test_cli_discovers_default_run_history_inside_input_dir(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    history_path = input_dir / lct.DEFAULT_TOML_PATH
    history_path.write_text(
        """
weight = 73.0

[analysis_windows.s01_d01_t01]
start_index = 526
end_index_exclusive = 60000
source_samples = 60000
""",
        encoding="utf-8",
    )
    calls = []

    def fake_process_stage(parent, initial_dir, config, output_dir, headless):
        calls.append(config)
        result = Path(output_dir) / "results_test"
        result.mkdir()
        return str(result)

    monkeypatch.setattr(lct, "run_process_stage", fake_process_stage)

    assert lct.main(["--input-dir", str(input_dir), "--step", "process"]) == 0
    effective = calls[0]
    assert effective["processing"]["participant_weight_kg"] == 73.0
    assert effective["processing"]["participant_weight_source"] == "toml"
    assert effective["analysis_windows"]["s01_d01_t01"]["start_index"] == 526
    assert effective["analysis_windows"]["s01_d01_t01"]["end_index_exclusive"] == 60000


def test_force_auto_ignores_even_an_existing_toml(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    config_path = tmp_path / "must-be-ignored.toml"
    config_path.write_text(
        """
weight = 12.0

[analysis_windows.s01_d01_t01]
start_index = 20
end_index_exclusive = 90
""",
        encoding="utf-8",
    )
    calls = []

    def fake_process_stage(parent, initial_dir, config, output_dir, headless):
        calls.append(config)
        result = Path(output_dir) / "results_test"
        result.mkdir()
        return str(result)

    monkeypatch.setattr(lct, "run_process_stage", fake_process_stage)

    assert (
        lct.main(
            [
                "--config",
                str(config_path),
                "--input-dir",
                str(input_dir),
                "--step",
                "process",
                "--force-auto",
            ]
        )
        == 0
    )
    effective = calls[0]
    assert effective["analysis_windows"] == {}
    assert effective["processing"]["participant_weight_kg"] == 70.0
    assert effective["processing"]["participant_weight_source"] == "fallback"
    assert effective["execution"]["force_auto"] is True


def test_missing_toml_warns_and_uses_automatic_defaults(tmp_path, monkeypatch, capsys):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    calls = []

    def fake_process_stage(parent, initial_dir, config, output_dir, headless):
        calls.append(config)
        result = Path(output_dir) / "results_test"
        result.mkdir()
        return str(result)

    monkeypatch.setattr(lct, "run_process_stage", fake_process_stage)
    missing = tmp_path / "does-not-exist.toml"

    assert (
        lct.main(
            [
                "--input-dir",
                str(input_dir),
                "--toml-path",
                str(missing),
                "--step",
                "process",
            ]
        )
        == 0
    )
    assert calls[0]["analysis_windows"] == {}
    terminal_output = capsys.readouterr().out
    assert "TOML not found" in terminal_output
    assert "Using built-in defaults and automatic analysis windows" in terminal_output


def test_cli_uses_subject_metadata_weight_when_weight_is_omitted(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "s01_d01_t01.csv").write_text("0,0,0,0,0\n", encoding="utf-8")
    (input_dir / "info_s01_d01.txt").write_text(
        "Subject,Day,Trial,BORG,Speed,Weight\nS01,01,T01,3,15,61.6\n",
        encoding="utf-8",
    )
    calls = []

    def fake_process_stage(parent, initial_dir, config, output_dir, headless):
        calls.append(config)
        result = Path(output_dir) / "results"
        result.mkdir()
        return str(result)

    monkeypatch.setattr(lct, "run_process_stage", fake_process_stage)

    assert lct.main(["-i", str(input_dir), "-s", "process", "-a"]) == 0
    assert calls[0]["processing"]["participant_weight_kg"] == 61.6
    assert calls[0]["processing"]["participant_weight_source"] == "metadata"


def test_rich_console_summary_and_milestones(capsys):
    """Test that rich display helpers execute without errors."""
    config = lct.get_default_unified_config()
    lct.print_treadmill_config_summary(config, input_dir="/tmp/test", mode="Test", step="filter")
    lct.print_pipeline_milestone(1, 5, "Test Milestone", "Details", "success")
    terminal_output = capsys.readouterr().out
    assert "Pipeline Step" in terminal_output
    assert "Median Window" in terminal_output
    assert "Max Comparison Methods" in terminal_output
    assert "Participant Weight Kg" in terminal_output
    assert "Test Milestone" in terminal_output


def test_help_documents_gui_cli_replay_and_uses_readable_command_contrast():
    help_dir = Path(lct.__file__).resolve().parent / "help"
    markdown = (help_dir / "treadmill_lc.md").read_text(encoding="utf-8")
    html = (help_dir / "treadmill_lc.html").read_text(encoding="utf-8")

    assert "Repeat a GUI run from the terminal" in markdown
    assert "Use the final banner" in markdown
    assert ">> vaila/treadmill_lc: Equivalent CLI" in markdown
    assert "analysis_windows.s01_d01_t01" in markdown
    assert "--force-auto" in markdown
    assert "first large impact peak" in markdown
    assert "uv run vaila/treadmill_lc.py" in markdown
    assert "python -m vaila.treadmill_lc" not in markdown
    assert "stable by default" in markdown
    assert "-T" in markdown
    assert "Body Weight" in markdown
    assert "--help" in markdown and "-h" in markdown

    assert "background: #f8fafc" in html
    assert "pre code { background: transparent; color: inherit" in html
    assert "pre { background: #0f172a" not in html
    assert "Use the final yellow banner" in html
    assert "Equivalent CLI for the completed GUI run" in html
    assert "--force-auto" in html
    assert "first large impact peak" in html
    assert "uv run vaila/treadmill_lc.py" in html
    assert "python -m vaila.treadmill_lc" not in html
    assert "stable <code>results</code>" in html
    assert "--timestamp-output" in html
    assert "Body Weight" in html
    assert "--help" in html and "-h" in html
