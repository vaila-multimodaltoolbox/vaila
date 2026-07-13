from __future__ import annotations

import os
from pathlib import Path

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
    t_limpo = np.array([0.0, 0.1, 0.2, 0.5, 0.6])
    t_reset = reset_times(t_limpo)
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

    def fake_clean_signal(file_path, parent=None):
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
    assert output_folder.name.startswith("clean_")
    assert (output_folder / "s01_d01_t01.csv").read_text() == "0,9,9,9,9\n"
    assert (output_folder / "s01_d01_t02.csv").read_text() == "0,5,6,7,8\n"
    assert (output_folder / "s01_d01_tara.csv").exists()
    assert not (output_folder / "s01_d01_t01_clean.csv").exists()


def test_is_calibration_file():
    assert is_calibration_file("s02_d01_tara.csv") is True
    assert is_calibration_file("s02_d01_peso.csv") is True
    assert is_calibration_file("s02_d01_10kg.csv") is True
    assert is_calibration_file("s02_d01_01kg.csv") is True
    assert is_calibration_file("s02_d01_t01.csv") is False
    assert is_calibration_file("s02_d01_t01_adjust_intervals.csv") is False
    assert is_calibration_file("borg_s02_d01.txt") is False


def test_get_group_weight_from_borg(tmp_path):
    borg_file = tmp_path / "borg_s01_d03.txt"
    content = (
        "Suj,Dia,Tent,Peso,BORG,PAS,PAD,FC,SpO2,FCOx,Vel\n"
        "S01,03,T01,61.4,2,138,72,124,96,131,15\n"
        "S01,03,T02*,61.4,2,135,78,112,97,111,15\n"
        "S01,03,T03*,61.4,3,132,83,134,97,120,15\n"
    )
    borg_file.write_text(content)

    assert get_group_weight_from_borg(str(borg_file)) == 61.4


def test_discover_calibration_and_borg(tmp_path):
    # Setup mock files
    (tmp_path / "s01_d03_tara.csv").touch()
    (tmp_path / "s01_d03_peso.csv").touch()
    (tmp_path / "s01_d03_05kg.csv").touch()
    (tmp_path / "s01_d03_10kg.csv").touch()
    (tmp_path / "borg_s01_d03.txt").touch()

    # Subdir
    subdir = tmp_path / "filtrado"
    subdir.mkdir()
    (subdir / "s01_d03_t01.csv").touch()

    tara, peso, plates, borg = discover_calibration_and_borg(str(subdir), "01", "03")

    assert tara == str(tmp_path / "s01_d03_tara.csv")
    assert peso == str(tmp_path / "s01_d03_peso.csv")
    assert len(plates) == 2
    assert str(tmp_path / "s01_d03_05kg.csv") in plates
    assert str(tmp_path / "s01_d03_10kg.csv") in plates
    assert borg == str(tmp_path / "borg_s01_d03.txt")


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
    tara = tmp_path / "s01_d01_tara.csv"
    peso = tmp_path / "s01_d01_peso.csv"
    pd.DataFrame(np.column_stack([np.arange(10), -np.ones((10, 4))])).to_csv(
        running, header=False, index=False
    )
    pd.DataFrame(np.column_stack([np.arange(10), np.zeros((10, 4))])).to_csv(
        tara, header=False, index=False
    )
    pd.DataFrame(np.column_stack([np.arange(10), -2 * np.ones((10, 4))])).to_csv(
        peso, header=False, index=False
    )

    def fail_filter(*args, **kwargs):
        raise AssertionError("processing filter should not be called")

    monkeypatch.setattr(lct, "butterworth_filter", fail_filter)

    grf_bw, grf_total, _, _ = load_data(
        str(running),
        str(tara),
        str(peso),
        peso_kg=80.0,
        apply_processing_filter=False,
    )

    assert grf_bw.shape == (10, 4)
    assert grf_total.shape == (10,)


def test_calculate_cop_system_uses_declared_cell_layout():
    # Cell order: 1 top-left, 2 bottom-left, 3 top-right, 4 bottom-right.
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
    dados = np.column_stack([np.arange(6, dtype=float) + col for col in range(4)])

    t_adj, dados_adj, records = apply_adjustment_intervals(t, dados, [(2, 4)], "nan")

    np.testing.assert_array_equal(t_adj, t)
    assert dados_adj.shape == dados.shape
    assert np.isnan(dados_adj[2:4, :]).all()
    np.testing.assert_array_equal(dados_adj[:2, :], dados[:2, :])
    np.testing.assert_array_equal(dados_adj[4:, :], dados[4:, :])
    assert records[0]["mode"] == "nan"
    assert records[0]["start_index"] == 2
    assert records[0]["end_index_exclusive"] == 4


def test_apply_adjustment_intervals_selected_cell_only():
    t = np.arange(6, dtype=float)
    dados = np.column_stack([np.arange(6, dtype=float) + col * 10 for col in range(4)])

    t_adj, dados_adj, records = apply_adjustment_intervals(
        t, dados, [{"start": 2, "end": 4, "cells": [2]}], "nan"
    )

    np.testing.assert_array_equal(t_adj, t)
    assert np.isnan(dados_adj[2:4, 2]).all()
    np.testing.assert_array_equal(dados_adj[:, 0], dados[:, 0])
    np.testing.assert_array_equal(dados_adj[:, 1], dados[:, 1])
    np.testing.assert_array_equal(dados_adj[:, 3], dados[:, 3])
    assert records[0]["cells_0based"] == [2]
    assert records[0]["cells_1based"] == [3]


def test_apply_adjustment_intervals_remove_shortens_signal():
    t = np.arange(6, dtype=float)
    dados = np.column_stack([np.arange(6, dtype=float) + col for col in range(4)])

    t_adj, dados_adj, records = apply_adjustment_intervals(t, dados, [(2, 4)], "remove")

    np.testing.assert_array_equal(t_adj, np.array([0.0, 1.0, 4.0, 5.0]))
    assert dados_adj.shape == (4, 4)
    assert records[0]["mode"] == "remove"
    assert records[0]["samples"] == 2


def test_apply_adjustment_intervals_zero_neutral_and_linear():
    t = np.arange(6, dtype=float)
    dados = np.column_stack([np.arange(6, dtype=float) for _ in range(4)])

    _, zeroed, _ = apply_adjustment_intervals(t, dados, [(2, 4)], "zero")
    assert np.all(zeroed[2:4, :] == 0.0)

    _, neutral, _ = apply_adjustment_intervals(t, dados, [(2, 4)], "neutral")
    # Boundary bridge between samples 1 and 4 gives [2, 3], neutral mean is 2.5.
    np.testing.assert_allclose(neutral[2:4, :], 2.5)

    _, linear, _ = apply_adjustment_intervals(t, dados, [(2, 4)], "linear")
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


def test_find_and_load_adjustment_metadata_for_limpo_file(tmp_path):
    limpo = tmp_path / "s01_d01_t01_LIMPO.csv"
    limpo.write_text("0,1,2,3,4\n")
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

    sidecar = find_adjustment_metadata_file(str(limpo))
    metadata = load_adjustment_metadata(str(limpo))

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
    limpo = tmp_path / "s01_d01_t01_LIMPO.csv"
    data = np.column_stack((np.arange(6, dtype=float), np.ones((6, 4))))
    np.savetxt(limpo, data, delimiter=",")
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
        str(limpo), get_default_interp_config(), root=None
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
    steps = [{"idx_start": 2, "idx_end": 8, "foot": "D"}]
    peaks = np.array([5])

    plot_trial_figures(grf_total, cop_x, cop_y, steps, peaks, "s01_d01_t01.csv", tmp_path)

    assert (tmp_path / "s01_d01_t01_processing_overview.png").exists()
    assert (tmp_path / "s01_d01_t01_processing_strike_attributes.png").exists()
    assert (tmp_path / "s01_d01_t01_processing_stride_map.png").exists()
    assert (tmp_path / "s01_d01_t01_processing_cop_trajectory.png").exists()
    report = tmp_path / "s01_d01_t01_processing_cop_report_interactive.html"
    assert report.exists()
    report_text = report.read_text(encoding="utf-8")
    assert "COP X - Medio-Lateral (cm)" in report_text
    assert "COP Y - Anterior-Posterior (cm)" in report_text
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
    tara = tmp_path / "s01_d01_tara.csv"
    peso = tmp_path / "s01_d01_peso.csv"

    # Write dummy CSV data
    pd.DataFrame(np.zeros((10, 5))).to_csv(trial, header=False, index=False)
    pd.DataFrame(np.zeros((10, 5))).to_csv(tara, header=False, index=False)
    pd.DataFrame(np.ones((10, 5))).to_csv(peso, header=False, index=False)

    # Save excluded metadata for this trial
    records = [{"start_index": 0, "end_index_exclusive": 10, "cells_0based": [0], "mode": "excluded"}]
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

    out_dir = lct.run_process_stage(parent=None, initial_dir=str(tmp_path))

    assert out_dir is not None
    steps_csv = Path(out_dir) / "s01_d01_t01_processing_steps.csv"
    assert not steps_csv.exists()

    metrics_csv = Path(out_dir) / "s01_d01_processing_metrics.csv"
    assert not metrics_csv.exists()

