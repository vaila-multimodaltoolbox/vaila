"""
Integration tests for vaila/vaila_and_jump.py — multi-function pipelines.

These tests verify that multiple functions work together correctly,
using the real sample data from tests/vaila_and_jump/.
No GUI, no Tkinter dialogs — only headless pipelines.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vaila.vaila_and_jump import (
    calc_fator_convert_mediapipe_simple,
    calculate_baseline,
    calculate_cg_frame,
    identify_jump_phases,
    process_jump_data,
)

# ──────────────────────────────────────────────
# Paths to real test data
# ──────────────────────────────────────────────
TEST_DATA = Path(__file__).resolve().parent / "vaila_and_jump"
TOF_CSV = TEST_DATA / "Time_of_flight_based_format" / "Time_of_flight1.csv"
HEIGHT_CSV = TEST_DATA / "Jump_height_based_format" / "Jump_height1.csv"
MEDIAPIPE_CSV = TEST_DATA / "vaila_mediapipe" / "salto_mp_norm_savgol.csv"


# ──────────────────────────────────────────────
# 1. Time-of-flight pipeline
# ──────────────────────────────────────────────
class TestTimeOfFlightPipeline:
    """End-to-end: CSV → process_jump_data → output CSV with correct metrics."""

    def test_produces_output_csv(self, tmp_path):
        process_jump_data(str(TOF_CSV), str(tmp_path), use_time_of_flight=True)
        csv_files = list(tmp_path.glob("*.csv"))
        assert len(csv_files) == 1, f"Expected 1 output CSV, got {len(csv_files)}"

    def test_output_has_correct_columns(self, tmp_path):
        process_jump_data(str(TOF_CSV), str(tmp_path), use_time_of_flight=True)
        csv_files = list(tmp_path.glob("*.csv"))
        df = pd.read_csv(csv_files[0])
        expected_cols = {
            "height_m",
            "velocity_m/s",
            "potential_energy_J",
            "kinetic_energy_J",
            "average_power_W",
            "relative_power_W/kg",
            "liftoff_force_N",
            "jump_performance_index",
            "total_time_s",
        }
        assert expected_cols.issubset(set(df.columns)), (
            f"Missing columns: {expected_cols - set(df.columns)}"
        )

    def test_output_has_correct_row_count(self, tmp_path):
        input_df = pd.read_csv(TOF_CSV)
        process_jump_data(str(TOF_CSV), str(tmp_path), use_time_of_flight=True)
        csv_files = list(tmp_path.glob("*.csv"))
        output_df = pd.read_csv(csv_files[0])
        assert len(output_df) == len(input_df)

    def test_height_values_are_positive(self, tmp_path):
        process_jump_data(str(TOF_CSV), str(tmp_path), use_time_of_flight=True)
        csv_files = list(tmp_path.glob("*.csv"))
        df = pd.read_csv(csv_files[0])
        assert (df["height_m"] > 0).all(), "All jump heights should be positive"

    def test_velocity_values_are_positive(self, tmp_path):
        process_jump_data(str(TOF_CSV), str(tmp_path), use_time_of_flight=True)
        csv_files = list(tmp_path.glob("*.csv"))
        df = pd.read_csv(csv_files[0])
        assert (df["velocity_m/s"] > 0).all()


# ──────────────────────────────────────────────
# 2. Jump-height pipeline
# ──────────────────────────────────────────────
class TestJumpHeightPipeline:
    """End-to-end: CSV → process_jump_data → output CSV with correct metrics."""

    def test_produces_output_csv(self, tmp_path):
        process_jump_data(str(HEIGHT_CSV), str(tmp_path), use_time_of_flight=False)
        csv_files = list(tmp_path.glob("*.csv"))
        assert len(csv_files) == 1

    def test_output_has_correct_columns(self, tmp_path):
        process_jump_data(str(HEIGHT_CSV), str(tmp_path), use_time_of_flight=False)
        csv_files = list(tmp_path.glob("*.csv"))
        df = pd.read_csv(csv_files[0])
        assert "height_m" in df.columns
        assert "velocity_m/s" in df.columns

    def test_height_matches_input(self, tmp_path):
        """For jump-height mode, output height should match input height."""
        input_df = pd.read_csv(HEIGHT_CSV)
        process_jump_data(str(HEIGHT_CSV), str(tmp_path), use_time_of_flight=False)
        csv_files = list(tmp_path.glob("*.csv"))
        output_df = pd.read_csv(csv_files[0])
        # Find height column in input (may be 'heigth_m' or 'height_m')
        height_col = next(
            (c for c in input_df.columns if "height" in c.lower() or "heigth" in c.lower()), None
        )
        if height_col:
            np.testing.assert_array_almost_equal(
                output_df["height_m"].values, input_df[height_col].values, decimal=2
            )


# ──────────────────────────────────────────────
# 3. MediaPipe CG calculation pipeline
# ──────────────────────────────────────────────
class TestMediaPipeCGPipeline:
    """Integration: load MediaPipe CSV → invert Y → compute factor → calculate CG."""

    @pytest.fixture
    def mediapipe_df(self):
        df = pd.read_csv(MEDIAPIPE_CSV)
        # Invert Y coords (same as process_mediapipe_data does)
        y_cols = [c for c in df.columns if c.endswith("_y")]
        for col in y_cols:
            df[col] = 1.0 - df[col]
        return df

    def test_conversion_factor_is_positive(self, mediapipe_df):
        factor = calc_fator_convert_mediapipe_simple(mediapipe_df, shank_length_real=0.40)
        assert factor > 0, "Conversion factor should be positive"

    def test_conversion_factor_is_reasonable(self, mediapipe_df):
        """Factor should convert normalized coords (~0-1) to meters.
        For a shank of 0.40m and normalized shank ~0.15-0.3, factor ~1.3-2.7."""
        factor = calc_fator_convert_mediapipe_simple(mediapipe_df, shank_length_real=0.40)
        assert 0.5 < factor < 10.0, f"Factor {factor} seems unreasonable"

    def test_cg_frame_adds_cg_columns(self, mediapipe_df):
        factor = calc_fator_convert_mediapipe_simple(mediapipe_df, shank_length_real=0.40)
        cg_x, cg_y = calculate_cg_frame(mediapipe_df, factor)
        assert len(cg_x) == len(mediapipe_df), "CG x should have same length as input"
        assert len(cg_y) == len(mediapipe_df), "CG y should have same length as input"

    def test_cg_values_are_finite(self, mediapipe_df):
        factor = calc_fator_convert_mediapipe_simple(mediapipe_df, shank_length_real=0.40)
        cg_x, cg_y = calculate_cg_frame(mediapipe_df, factor)
        assert np.all(np.isfinite(cg_x)), "CG x values should be finite"
        assert np.all(np.isfinite(cg_y)), "CG y values should be finite"


# ──────────────────────────────────────────────
# 4. MediaPipe baseline + jump phases pipeline
# ──────────────────────────────────────────────
class TestMediaPipeJumpPhasesPipeline:
    """Integration: MediaPipe CSV → CG → baseline → identify jump phases."""

    @pytest.fixture
    def prepared_data(self):
        """Load MediaPipe data and prepare it through the full pipeline up to phase detection."""
        df = pd.read_csv(MEDIAPIPE_CSV)
        # Invert Y coords
        y_cols = [c for c in df.columns if c.endswith("_y")]
        for col in y_cols:
            df[col] = 1.0 - df[col]
        # Calculate CG
        factor = calc_fator_convert_mediapipe_simple(df, shank_length_real=0.40)
        cg_x, cg_y = calculate_cg_frame(df, factor)
        df["cg_x"] = cg_x
        df["cg_y"] = cg_y
        # Add meters columns for feet
        df["right_foot_index_y_m"] = df["right_foot_index_y"] * factor
        df["left_foot_index_y_m"] = df["left_foot_index_y"] * factor
        # Calculate baseline
        feet_baseline, cg_baseline = calculate_baseline(df, n_frames=10)
        # Normalize CG
        df["cg_y_normalized"] = df["cg_y"] - cg_baseline
        return df, feet_baseline, cg_baseline

    def test_baseline_values_are_reasonable(self, prepared_data):
        _df, feet_bl, cg_bl = prepared_data
        assert feet_bl > 0, "Feet baseline should be positive"
        assert cg_bl > 0, "CG baseline should be positive"

    def test_identify_jump_phases_returns_expected_keys(self, prepared_data):
        df, feet_bl, cg_bl = prepared_data
        results = identify_jump_phases(df, feet_bl, cg_bl, fps=240)
        expected_keys = {
            "takeoff_frame",
            "max_height_frame",
            "landing_frame",
            "flight_time_s",
            "max_height_m",
            "propulsion_start_frame",
            "height_cg_method_m",
            "squat_depth_m",
            "height_flight_time_method_m",
        }
        assert expected_keys.issubset(set(results.keys())), (
            f"Missing keys: {expected_keys - set(results.keys())}"
        )

    def test_phase_frames_are_in_order(self, prepared_data):
        df, feet_bl, cg_bl = prepared_data
        results = identify_jump_phases(df, feet_bl, cg_bl, fps=240)
        squat = results["propulsion_start_frame"]
        takeoff = results["takeoff_frame"]
        peak = results["max_height_frame"]
        landing = results["landing_frame"]
        # Squat should come before or at takeoff; takeoff before peak; peak before landing
        assert squat <= takeoff, f"Squat ({squat}) should be <= takeoff ({takeoff})"
        assert takeoff <= peak, f"Takeoff ({takeoff}) should be <= peak ({peak})"
        assert peak <= landing, f"Peak ({peak}) should be <= landing ({landing})"

    def test_flight_time_is_positive(self, prepared_data):
        df, feet_bl, cg_bl = prepared_data
        results = identify_jump_phases(df, feet_bl, cg_bl, fps=240)
        assert results["flight_time_s"] >= 0, "Flight time should be non-negative"

    def test_max_height_is_positive(self, prepared_data):
        df, feet_bl, cg_bl = prepared_data
        results = identify_jump_phases(df, feet_bl, cg_bl, fps=240)
        assert results["max_height_m"] > 0, "Max height should be positive"

    def test_squat_depth_is_positive(self, prepared_data):
        df, feet_bl, cg_bl = prepared_data
        results = identify_jump_phases(df, feet_bl, cg_bl, fps=240)
        assert results["squat_depth_m"] >= 0, "Squat depth should be non-negative"


# ──────────────────────────────────────────────
# 5. Cross-pipeline consistency
# ──────────────────────────────────────────────
class TestCrossPipelineConsistency:
    """Verify that different data formats produce consistent results."""

    def test_tof_and_height_files_produce_similar_structure(self, tmp_path):
        """Both pipelines should produce output CSVs with the same column structure."""
        tof_dir = tmp_path / "tof"
        height_dir = tmp_path / "height"
        tof_dir.mkdir()
        height_dir.mkdir()

        process_jump_data(str(TOF_CSV), str(tof_dir), use_time_of_flight=True)
        process_jump_data(str(HEIGHT_CSV), str(height_dir), use_time_of_flight=False)

        tof_files = list(tof_dir.glob("*.csv"))
        height_files = list(height_dir.glob("*.csv"))

        assert len(tof_files) == 1
        assert len(height_files) == 1

        tof_df = pd.read_csv(tof_files[0])
        height_df = pd.read_csv(height_files[0])

        # Both should have the same columns
        assert set(tof_df.columns) == set(height_df.columns), (
            "Both pipelines should produce the same output columns"
        )

    def test_second_tof_file_also_works(self, tmp_path):
        """Verify the second test file also processes successfully."""
        tof2 = TEST_DATA / "Time_of_flight_based_format" / "Time_of_flight2.csv"
        if tof2.exists():
            process_jump_data(str(tof2), str(tmp_path), use_time_of_flight=True)
            csv_files = list(tmp_path.glob("*.csv"))
            assert len(csv_files) == 1

    def test_second_height_file_also_works(self, tmp_path):
        """Verify the second test file also processes successfully."""
        h2 = TEST_DATA / "Jump_height_based_format" / "Jump_height2.csv"
        if h2.exists():
            process_jump_data(str(h2), str(tmp_path), use_time_of_flight=False)
            csv_files = list(tmp_path.glob("*.csv"))
            assert len(csv_files) == 1
