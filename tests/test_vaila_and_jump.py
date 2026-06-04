"""
Unit tests for vaila/vaila_and_jump.py — pure calculation functions.

Update Date: 03 June 2026
Version: 0.3.47

These tests verify each atomic calculation function in isolation,
using known inputs and expected outputs based on physics formulas.
No GUI, no I/O, no external dependencies beyond numpy/pandas.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure the project root is on sys.path so we can import vaila.*
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vaila.vaila_and_jump import (
    _cmj_height_quality_check,
    _iter_jump_data_dirs,
    _jump_context_from_cfg,
    _load_jump_context_from_file,
    _load_jump_context_from_toml,
    _looks_like_vaila_output_csv,
    _parse_locale_float,
    _team_summary_table,
    calculate_average_power,
    calculate_baseline,
    calculate_force,
    calculate_jump_height,
    calculate_kinematics,
    calculate_kinetic_energy,
    calculate_liftoff_force,
    calculate_potential_energy,
    calculate_power,
    calculate_time_of_flight,
    calculate_velocity,
    identify_jump_phases,
)


# ──────────────────────────────────────────────
# 1. calculate_force
# ──────────────────────────────────────────────
class TestCalculateForce:
    def test_default_gravity(self):
        assert calculate_force(75.0) == pytest.approx(75.0 * 9.81)

    def test_custom_gravity(self):
        assert calculate_force(80.0, gravity=10.0) == pytest.approx(800.0)

    def test_zero_mass(self):
        assert calculate_force(0) == pytest.approx(0.0)

    def test_negative_mass(self):
        # Physics edge case — function should still compute m*g
        assert calculate_force(-50.0) == pytest.approx(-50.0 * 9.81)


# ──────────────────────────────────────────────
# 2. calculate_jump_height
# ──────────────────────────────────────────────
class TestCalculateJumpHeight:
    def test_known_value(self):
        # h = g * t^2 / 8;  t=0.45s => h = 9.81*0.2025/8 = 0.24826...
        expected = 9.81 * 0.45**2 / 8
        assert calculate_jump_height(0.45) == pytest.approx(expected)

    def test_zero_flight(self):
        assert calculate_jump_height(0) == pytest.approx(0.0)

    def test_custom_gravity(self):
        expected = 10.0 * 0.5**2 / 8
        assert calculate_jump_height(0.5, gravity=10.0) == pytest.approx(expected)


# ──────────────────────────────────────────────
# 3. calculate_velocity
# ──────────────────────────────────────────────
class TestCalculateVelocity:
    def test_known_height(self):
        h = 0.25
        expected = math.sqrt(2 * 9.81 * h)
        assert calculate_velocity(h) == pytest.approx(expected)

    def test_zero_height(self):
        assert calculate_velocity(0) == pytest.approx(0.0)

    def test_custom_gravity(self):
        h = 0.30
        g = 10.0
        expected = math.sqrt(2 * g * h)
        assert calculate_velocity(h, gravity=g) == pytest.approx(expected)


# ──────────────────────────────────────────────
# 4. calculate_kinetic_energy
# ──────────────────────────────────────────────
class TestCalculateKineticEnergy:
    def test_known_values(self):
        # KE = 0.5 * m * v^2
        assert calculate_kinetic_energy(75.0, 2.21) == pytest.approx(0.5 * 75.0 * 2.21**2)

    def test_zero_velocity(self):
        assert calculate_kinetic_energy(75.0, 0) == pytest.approx(0.0)

    def test_zero_mass(self):
        assert calculate_kinetic_energy(0, 5.0) == pytest.approx(0.0)


# ──────────────────────────────────────────────
# 5. calculate_potential_energy
# ──────────────────────────────────────────────
class TestCalculatePotentialEnergy:
    def test_known_values(self):
        # PE = m * g * h
        assert calculate_potential_energy(75.0, 0.25) == pytest.approx(75.0 * 9.81 * 0.25)

    def test_zero_height(self):
        assert calculate_potential_energy(75.0, 0) == pytest.approx(0.0)

    def test_custom_gravity(self):
        assert calculate_potential_energy(80.0, 0.30, gravity=10.0) == pytest.approx(
            80.0 * 10.0 * 0.30
        )


# ──────────────────────────────────────────────
# 6. calculate_average_power
# ──────────────────────────────────────────────
class TestCalculateAveragePower:
    def test_known_values(self):
        pe = 184.0
        ct = 0.22
        assert calculate_average_power(pe, ct) == pytest.approx(pe / ct)

    def test_small_contact_time(self):
        pe = 100.0
        ct = 0.01
        assert calculate_average_power(pe, ct) == pytest.approx(10000.0)


# ──────────────────────────────────────────────
# 7. calculate_power
# ──────────────────────────────────────────────
class TestCalculatePower:
    def test_known_values(self):
        force = 735.75
        height = 0.25
        ct = 0.22
        expected = (force * height) / ct
        assert calculate_power(force, height, ct) == pytest.approx(expected)


# ──────────────────────────────────────────────
# 8. calculate_liftoff_force
# ──────────────────────────────────────────────
class TestCalculateLiftoffForce:
    def test_known_values(self):
        m, v, ct = 75.0, 2.21, 0.22
        expected = m * 9.81 + (m * v) / ct
        assert calculate_liftoff_force(m, v, ct) == pytest.approx(expected)

    def test_custom_gravity(self):
        m, v, ct, g = 80.0, 2.5, 0.25, 10.0
        expected = m * g + (m * v) / ct
        assert calculate_liftoff_force(m, v, ct, gravity=g) == pytest.approx(expected)


# ──────────────────────────────────────────────
# 9. calculate_time_of_flight
# ──────────────────────────────────────────────
class TestCalculateTimeOfFlight:
    def test_known_height(self):
        h = 0.25
        expected = math.sqrt(8 * h / 9.81)
        assert calculate_time_of_flight(h) == pytest.approx(expected)

    def test_zero_height(self):
        assert calculate_time_of_flight(0) == pytest.approx(0.0)

    def test_roundtrip_with_jump_height(self):
        """height → time_of_flight → height should give back original height."""
        original_h = 0.30
        tof = calculate_time_of_flight(original_h)
        recovered_h = calculate_jump_height(tof)
        assert recovered_h == pytest.approx(original_h, rel=1e-9)


# ──────────────────────────────────────────────
# 10. TOML context loading
# ──────────────────────────────────────────────
class TestParseLocaleFloat:
    def test_dot_decimal(self):
        assert _parse_locale_float("0.40") == pytest.approx(0.40)

    def test_comma_decimal(self):
        assert _parse_locale_float("0,42") == pytest.approx(0.42)

    def test_leading_dot(self):
        assert _parse_locale_float(".40") == pytest.approx(0.40)

    def test_numeric_input(self):
        assert _parse_locale_float(0.38) == pytest.approx(0.38)


class TestTomlContextLoading:
    def _write_toml(self, path: Path, mass: float, fps: float, shank: float):
        content = f"[jump_context]\nmass_kg = {mass}\nfps = {fps}\nshank_length_m = {shank}\n"
        path.write_text(content, encoding="utf-8")

    def test_load_from_file_valid(self, tmp_path):
        toml_path = tmp_path / "config.toml"
        self._write_toml(toml_path, 75.0, 240.0, 0.40)
        ctx = _load_jump_context_from_file(toml_path)
        assert ctx is not None
        assert ctx["mass_kg"] == pytest.approx(75.0)
        assert ctx["fps"] == pytest.approx(240.0)
        assert ctx["shank_length_m"] == pytest.approx(0.40)

    def test_jump_context_from_cfg_comma_string_shank(self):
        ctx = _jump_context_from_cfg({"mass_kg": 75.0, "fps": 240.0, "shank_length_m": "0,43"})
        assert ctx is not None
        assert ctx["shank_length_m"] == pytest.approx(0.43)

    def test_load_from_file_missing(self):
        ctx = _load_jump_context_from_file("/nonexistent/path.toml")
        assert ctx is None

    def test_load_from_file_invalid_values(self, tmp_path):
        toml_path = tmp_path / "config.toml"
        self._write_toml(toml_path, 0, 0, 0)  # all zeros → should return None
        ctx = _load_jump_context_from_file(toml_path)
        assert ctx is None

    def test_load_from_toml_directory(self, tmp_path):
        toml_path = tmp_path / "vaila_and_jump_config.toml"
        self._write_toml(toml_path, 80.0, 120.0, 0.38)
        ctx = _load_jump_context_from_toml(base_dir=tmp_path)
        assert ctx is not None
        assert ctx["mass_kg"] == pytest.approx(80.0)

    def test_load_from_toml_no_file(self, tmp_path):
        # tmp_path has no config file → falls through to package dir
        # which also shouldn't have one → returns None
        ctx = _load_jump_context_from_toml(base_dir=tmp_path)
        # May return None or find a config in the package dir;
        # we just verify it doesn't crash
        assert ctx is None or isinstance(ctx, dict)


# ──────────────────────────────────────────────
# 11. calculate_baseline
# ──────────────────────────────────────────────
class TestCalculateBaseline:
    def test_with_synthetic_data(self):
        n = 20
        data = pd.DataFrame(
            {
                "right_foot_index_y": np.full(n, 0.85),
                "left_foot_index_y": np.full(n, 0.87),
                "cg_y": np.full(n, 0.50),
            }
        )
        feet_bl, cg_bl = calculate_baseline(data, n_frames=10)
        assert feet_bl == pytest.approx((0.85 + 0.87) / 2)
        assert cg_bl == pytest.approx(0.50)

    def test_with_varying_data(self):
        n = 20
        rng = np.random.default_rng(42)
        rf = rng.normal(0.85, 0.01, n)
        lf = rng.normal(0.87, 0.01, n)
        cg = rng.normal(0.50, 0.01, n)
        data = pd.DataFrame(
            {
                "right_foot_index_y": rf,
                "left_foot_index_y": lf,
                "cg_y": cg,
            }
        )
        feet_bl, cg_bl = calculate_baseline(data, n_frames=10)
        expected_feet = (rf[:10].mean() + lf[:10].mean()) / 2
        assert feet_bl == pytest.approx(expected_feet, rel=1e-6)
        assert cg_bl == pytest.approx(cg[:10].mean(), rel=1e-6)


class TestIdentifyJumpPhases:
    @staticmethod
    def _cmj_with_late_landing_dip():
        cg = np.zeros(120, dtype=float)
        cg[20:36] = np.linspace(0.0, -0.22, 16)
        cg[36:66] = np.linspace(-0.22, 0.52, 30)
        cg[66:90] = np.linspace(0.52, 0.0, 24)
        cg[90:120] = np.linspace(0.0, -0.55, 30)
        return pd.DataFrame({"cg_y_normalized": cg})

    def test_propulsion_is_anchored_before_com_peak(self):
        data = self._cmj_with_late_landing_dip()
        results = identify_jump_phases(data, feet_baseline=0.0, _cg_baseline=0.0, fps=60)

        assert results["propulsion_start_frame"] < results["max_height_frame"]
        assert results["takeoff_frame"] <= results["max_height_frame"]
        assert results["propulsion_start_frame"] < 90
        assert results["ascent_time_s"] > 0

    def test_single_frame_initial_spike_does_not_define_propulsion(self):
        data = self._cmj_with_late_landing_dip()
        data.loc[5, "cg_y_normalized"] = -0.70
        results = identify_jump_phases(data, feet_baseline=0.0, _cg_baseline=0.0, fps=60)

        assert 20 <= results["propulsion_start_frame"] <= 40
        assert results["takeoff_frame"] <= results["max_height_frame"]


class TestCalculateKinematics:
    @staticmethod
    def _valid_lower_limb_data():
        row = {
            "left_hip_x": -1.0,
            "left_hip_y": 2.0,
            "left_knee_x": -0.9,
            "left_knee_y": 1.0,
            "left_ankle_x": -1.2,
            "left_ankle_y": 0.1,
            "right_hip_x": 1.0,
            "right_hip_y": 2.0,
            "right_knee_x": 0.9,
            "right_knee_y": 1.0,
            "right_ankle_x": 1.2,
            "right_ankle_y": 0.1,
            "cg_x": 0.0,
            "cg_y": 1.0,
        }
        return pd.DataFrame([row, row, row])

    def test_squat_fppa_uses_propulsion_start_frame(self):
        data = self._valid_lower_limb_data()
        results = {"fps": 60, "propulsion_start_frame": 1, "landing_frame": 2}

        kinematics = calculate_kinematics(data, results)

        assert kinematics["fppa_left_squat_deg"] is not None
        assert kinematics["fppa_right_squat_deg"] is not None

    def test_squat_fppa_accepts_legacy_squat_frame_alias(self):
        data = self._valid_lower_limb_data()
        results = {"fps": 60, "squat_frame": 1, "landing_frame": 2}

        kinematics = calculate_kinematics(data, results)

        assert kinematics["fppa_left_squat_deg"] is not None
        assert kinematics["fppa_right_squat_deg"] is not None


class TestCmjHeightQualityCheck:
    def test_probable_error_uses_foot_contact_correction(self):
        phase_results = {
            "height_cg_method_m": 1.000291,
            "landing_frame": 78,
            "left_takeoff_frame": 33,
            "right_takeoff_frame": 34,
            "left_landing_frame": 70,
            "right_landing_frame": 69,
        }

        qc = _cmj_height_quality_check(phase_results, fps=59.627)

        assert qc["height_qc_status"] == "probable_error"
        assert qc["height_qc_correction_applied"] is True
        assert qc["height_qc_recommended_source"] == "foot_contact_flight_time"
        assert qc["height_qc_recommended_m"] == pytest.approx(0.423, abs=0.002)
        assert qc["takeoff_frame_foot_contact"] == 34
        assert qc["landing_frame_foot_contact"] == 69

    def test_high_but_plausible_height_is_not_corrected(self):
        phase_results = {
            "height_cg_method_m": 0.70,
            "left_takeoff_frame": 10,
            "right_takeoff_frame": 10,
            "left_landing_frame": 40,
            "right_landing_frame": 40,
        }

        qc = _cmj_height_quality_check(phase_results, fps=60)

        assert qc["height_qc_status"] == "very_high_plausible"
        assert qc["height_qc_correction_applied"] is False
        assert qc["height_qc_recommended_m"] == pytest.approx(0.70)


# ──────────────────────────────────────────────
# 12. Edge cases / consistency checks
# ──────────────────────────────────────────────
class TestEdgeCases:
    def test_energy_conservation(self):
        """At peak height, KE at takeoff ≈ PE at peak (energy conservation)."""
        m = 75.0
        h = 0.25
        v = calculate_velocity(h)
        ke = calculate_kinetic_energy(m, v)
        pe = calculate_potential_energy(m, h)
        assert ke == pytest.approx(pe, rel=1e-9)

    def test_time_of_flight_symmetry(self):
        """Ascending time should equal descending time in ballistic flight."""
        h = 0.30
        tof = calculate_time_of_flight(h)
        # Max height reached at tof/2
        v0 = calculate_velocity(h)
        t_up = v0 / 9.81
        assert t_up == pytest.approx(tof / 2, rel=1e-6)


# ──────────────────────────────────────────────
# 13. Team batch helpers
# ──────────────────────────────────────────────
class TestTeamBatchHelpers:
    def test_looks_like_vaila_output_csv(self):
        assert _looks_like_vaila_output_csv(Path("S01_jump_results_20260101_000000.csv"))
        assert _looks_like_vaila_output_csv(Path("S01_calibrated_20260101_000000.csv"))
        assert _looks_like_vaila_output_csv(Path("S01_jump_timeseries_20260101.csv"))
        assert not _looks_like_vaila_output_csv(Path("salto_mp_norm_savgol.csv"))

    def test_iter_jump_data_dirs_finds_only_configured_folders(self, tmp_path):
        # Athlete folder with TOML + raw CSV → should be found
        a = tmp_path / "S01"
        a.mkdir()
        (a / "S01.csv").write_text("frame_index\n0\n")
        (a / "vaila_and_jump_config.toml").write_text(
            "[jump_context]\nmass_kg = 70\nfps = 30\nshank_length_m = 0.4\n"
        )
        # Folder with CSV but no TOML → ignored
        b = tmp_path / "no_config"
        b.mkdir()
        (b / "data.csv").write_text("frame_index\n0\n")
        # Folder with TOML but only an output CSV → ignored
        c = tmp_path / "only_outputs"
        c.mkdir()
        (c / "vaila_and_jump_config.toml").write_text(
            "[jump_context]\nmass_kg = 70\nfps = 30\nshank_length_m = 0.4\n"
        )
        (c / "x_jump_results_20260101_000000.csv").write_text("a\n1\n")
        # A vailá output directory should be pruned from the walk
        out = tmp_path / "vaila_team_jump_20260101_000000" / "S01"
        out.mkdir(parents=True)
        (out / "vaila_and_jump_config.toml").write_text(
            "[jump_context]\nmass_kg = 70\nfps = 30\nshank_length_m = 0.4\n"
        )
        (out / "S01.csv").write_text("frame_index\n0\n")

        found = _iter_jump_data_dirs(tmp_path)
        assert found == [a]

    def test_team_summary_table_stats(self):
        df = pd.DataFrame(
            [
                {"height_cg_method_m": 0.30, "mass_kg": 70.0},
                {"height_cg_method_m": 0.40, "mass_kg": 80.0},
            ]
        )
        summary = _team_summary_table(df)
        height_row = summary[summary["metric"] == "Jump height (CG raw) [m]"].iloc[0]
        assert int(height_row["n"]) == 2
        assert height_row["mean"] == pytest.approx(0.35)
        assert height_row["min"] == pytest.approx(0.30)
        assert height_row["max"] == pytest.approx(0.40)
