"""
Unit tests for vaila/vaila_and_jump.py — pure calculation functions.

These tests verify each atomic calculation function in isolation,
using known inputs and expected outputs based on physics formulas.
No GUI, no I/O, no external dependencies beyond numpy/pandas.
"""

import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure the project root is on sys.path so we can import vaila.*
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vaila.vaila_and_jump import (
    _load_jump_context_from_file,
    _load_jump_context_from_toml,
    calculate_average_power,
    calculate_baseline,
    calculate_force,
    calculate_jump_height,
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
class TestTomlContextLoading:
    def _write_toml(self, path: Path, mass: float, fps: float, shank: float):
        content = (
            "[jump_context]\n"
            f"mass_kg = {mass}\n"
            f"fps = {fps}\n"
            f"shank_length_m = {shank}\n"
        )
        path.write_text(content, encoding="utf-8")

    def test_load_from_file_valid(self, tmp_path):
        toml_path = tmp_path / "config.toml"
        self._write_toml(toml_path, 75.0, 240.0, 0.40)
        ctx = _load_jump_context_from_file(toml_path)
        assert ctx is not None
        assert ctx["mass_kg"] == pytest.approx(75.0)
        assert ctx["fps"] == pytest.approx(240.0)
        assert ctx["shank_length_m"] == pytest.approx(0.40)

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
