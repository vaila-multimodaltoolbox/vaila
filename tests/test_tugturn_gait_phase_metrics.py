"""Gait-window cadence and speed in ``vaila.tugturn``.

The global ``Cadence_steps_per_min`` / ``Velocity_m_s`` span the first to the last heel strike
of the whole TUG trial, so the pause before the turn and the turn sit in the denominator.
``gait_phase_cadence_velocity`` / ``gait_phase_metrics`` restrict both to each straight-gait
window. Known-value fixtures: heel strikes every 0.5 s (120 steps/min) and a CoM moving at a
constant horizontal speed.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from vaila.tugturn import gait_phase_cadence_velocity, gait_phase_metrics

FPS = 100.0


def _straight_com(n_frames: int, speed_m_s: float, fps: float = FPS) -> np.ndarray:
    t = np.arange(n_frames) / fps
    return np.column_stack([np.zeros_like(t), speed_m_s * t])


def test_constant_step_rate_and_speed():
    times = np.arange(1.0, 6.0 + 1e-9, 0.5)  # 11 heel strikes, 10 intervals over 5 s
    com = _straight_com(1000, 1.2)
    out = gait_phase_cadence_velocity(times, com, FPS)
    assert out["n_heel_strikes"] == 11
    assert out["span_s"] == pytest.approx(5.0)
    assert out["cadence_steps_per_min"] == pytest.approx(120.0)
    assert out["path_m"] == pytest.approx(6.0)
    assert out["velocity_m_s"] == pytest.approx(1.2)


def test_unsorted_times_are_sorted():
    times = [3.0, 1.0, 2.0]
    out = gait_phase_cadence_velocity(times, _straight_com(500, 1.0), FPS)
    assert out["cadence_steps_per_min"] == pytest.approx(60.0)
    assert out["velocity_m_s"] == pytest.approx(1.0)


def test_speed_uses_horizontal_path_not_displacement():
    # Lateral zig-zag of +/-0.05 m every frame on top of 1 m/s forward progression
    com = _straight_com(400, 1.0)
    com[:, 0] = np.where(np.arange(400) % 2 == 0, 0.05, -0.05)
    out = gait_phase_cadence_velocity([0.0, 1.0, 2.0], com, FPS)
    per_frame = math.hypot(0.1, 0.01)
    assert out["path_m"] == pytest.approx(200 * per_frame)
    assert out["velocity_m_s"] > 1.0


@pytest.mark.parametrize("times", [[], [2.0], [2.0, 2.0]])
def test_fewer_than_two_distinct_strikes_give_nan(times):
    out = gait_phase_cadence_velocity(times, _straight_com(500, 1.0), FPS)
    assert math.isnan(out["cadence_steps_per_min"])
    assert math.isnan(out["velocity_m_s"])


def test_nan_in_com_window_gives_nan_speed_but_keeps_cadence():
    com = _straight_com(500, 1.0)
    com[150, 1] = np.nan
    out = gait_phase_cadence_velocity([1.0, 1.5, 2.0], com, FPS)
    assert out["cadence_steps_per_min"] == pytest.approx(120.0)
    assert math.isnan(out["velocity_m_s"])


def test_missing_com_keeps_cadence():
    out = gait_phase_cadence_velocity([1.0, 1.5, 2.0], None, FPS)
    assert out["cadence_steps_per_min"] == pytest.approx(120.0)
    assert math.isnan(out["velocity_m_s"])


def _steps(times, phase):
    return [{"Time_s": t, "Phase": phase} for t in times]


def test_phase_metrics_exclude_the_pause_and_turn():
    """Two 4-s gait windows at 120 steps/min separated by a 10-s pause/turn.

    A first-to-last heel-strike rate over the whole trial would give
    (n_HS) / 18 s; the gait-window rate must stay at 120 steps/min.
    """
    first = np.arange(2.0, 6.0 + 1e-9, 0.5)  # 9 HS
    second = np.arange(16.0, 20.0 + 1e-9, 0.5)  # 9 HS
    steps = _steps(first, "first_gait") + _steps(second, "second_gait")
    steps += _steps([8.0, 12.0], "Other")  # must be ignored
    com = _straight_com(2500, 0.8)
    g = gait_phase_metrics(steps, com, FPS)
    assert g["Cadence_First_Gait_steps_per_min"] == pytest.approx(120.0)
    assert g["Cadence_Second_Gait_steps_per_min"] == pytest.approx(120.0)
    assert g["Cadence_Gait_steps_per_min"] == pytest.approx(120.0)
    assert g["Velocity_First_Gait_m_s"] == pytest.approx(0.8)
    assert g["Velocity_Gait_m_s"] == pytest.approx(0.8)
    whole_trial_rate = (len(first) + len(second)) / ((second[-1] - first[0]) / 60.0)
    assert whole_trial_rate < 70.0  # what the global definition would report


def test_pooled_values_weight_phases_by_duration():
    first = np.arange(0.0, 4.0 + 1e-9, 0.5)  # 8 intervals in 4 s -> 120 steps/min
    second = np.arange(10.0, 12.0 + 1e-9, 1.0)  # 2 intervals in 2 s -> 60 steps/min
    steps = _steps(first, "first_gait") + _steps(second, "second_gait")
    g = gait_phase_metrics(steps, None, FPS)
    assert g["Cadence_First_Gait_steps_per_min"] == pytest.approx(120.0)
    assert g["Cadence_Second_Gait_steps_per_min"] == pytest.approx(60.0)
    assert g["Cadence_Gait_steps_per_min"] == pytest.approx(10 / 6 * 60.0)
    assert g["Velocity_Gait_m_s"] is None


def test_missing_phase_gives_none_and_valid_json():
    steps = _steps(np.arange(1.0, 3.0 + 1e-9, 0.5), "first_gait")
    g = gait_phase_metrics(steps, _straight_com(500, 1.0), FPS)
    assert g["Cadence_Second_Gait_steps_per_min"] is None
    assert g["Velocity_Second_Gait_m_s"] is None
    assert g["Cadence_Gait_steps_per_min"] == pytest.approx(120.0)
    json.dumps(g, allow_nan=False)
