"""Keeper response, goal crossing and the TV speed figure of the FreeKiki shot package."""

from __future__ import annotations

import math

import numpy as np
import pytest

from vaila import freekiki_shot_tv as st
from vaila.freekiki_pass import speed_headline_kmh


def _keeper(n: int = 15, rise_at: int = 6) -> dict[int, dict[str, np.ndarray]]:
    poses = {}
    for f in range(n):
        z = 0.7 + (0.0 if f < rise_at else 0.35 + 0.1 * (f - rise_at))
        kp = {
            "left-ankle": np.array([-49.0, -4.6, 0.05]),
            "right-ankle": np.array([-49.0, -4.4, 0.05]),
        }
        kp["left-hip"] = np.array([-49.0, -4.6, 0.9])
        kp["right-hip"] = np.array([-49.0, -4.4, 0.9])
        kp["right-wrist"] = np.array([-49.0, -4.8, z])
        for tip in st.HAND_TIPS:
            kp[f"right-{tip}-tip"] = np.array([-49.0, -4.9, z])
            kp[f"left-{tip}-tip"] = np.array([-49.0, -4.2, 0.6])
        kp["left-wrist"] = np.array([-49.0, -4.25, 0.6])
        poses[f] = kp
    return poses


def _ball(n: int = 15) -> np.ndarray:
    t = np.arange(n)
    x = -42.0 - 0.75 * t
    y = -10.8 + 0.42 * t
    z = 0.11 + 0.25 * t - 0.012 * t**2
    return np.column_stack([x, y, z])


def test_keeper_response_finds_the_onset_and_the_closest_hand() -> None:
    fps = 24.0
    ball = _ball()
    r = st.keeper_response(_keeper(rise_at=6), ball, fps)
    assert r["reaching_hand"] == "right"
    assert r["onset_frame"] == 6
    assert math.isclose(r["onset_s"], 6 / fps)
    gaps = [row["hand_gap_m"] for row in r["series"]]
    assert r["closest_frame"] == int(np.argmin(gaps))
    assert math.isclose(r["closest_gap_m"], min(gaps))
    assert r["motor_time_after_onset_s"] == pytest.approx(r["closest_s"] - r["onset_s"])


def test_keeper_response_needs_frames() -> None:
    with pytest.raises(ValueError, match="fewer than four"):
        st.keeper_response({0: {}}, _ball(1), 24.0)


def test_goal_crossing_reports_the_post_side_and_distance() -> None:
    ball = np.array([[-51.0, -4.6, 1.8], [-52.0, -4.2, 1.7]])
    c = st.goal_crossing(ball, -52.45)
    assert c["reaches_line"] is True
    assert c["y_at_line_m"] == pytest.approx(-4.02)
    assert c["outside_post_m"] == pytest.approx(4.02 - 3.66)
    assert c["inside_goal"] is False
    inside = st.goal_crossing(np.array([[-51.0, -2.0, 1.0], [-52.0, -2.0, 1.0]]), -52.45)
    assert inside["inside_goal"] is True
    away = st.goal_crossing(np.array([[-52.0, 0.0, 1.0], [-51.0, 0.0, 1.0]]), -52.45)
    assert away["reaches_line"] is False


def test_tv_speed_is_the_start_value_rounded_to_five() -> None:
    assert speed_headline_kmh(92.3, 76.6) == 90
    assert speed_headline_kmh(None, 77.0) == 75  # no start value: the mean
    assert speed_headline_kmh(None, None) is None


def test_displayed_shot_speed_never_rises_after_the_kick() -> None:
    fps = 24.0
    raw = [0.0, 17.9, 24.78, 24.12, 23.49, 22.9, 23.0, 22.1]  # kick inside the first step
    v = st.monotone_flight_speed(raw, fps, kick_time_s=0.012, launch_mps=25.6)
    assert v[0] is None  # before the kick
    after = [x for x in v if x is not None]
    assert after[0] < 25.6 and after[0] > 24.78  # between the launch and the first full frame
    assert all(b <= a for a, b in zip(after, after[1:], strict=False))


def test_gap_for_tv_is_rounded_to_ten_centimetres() -> None:
    assert st._gap_cm(0.67) == 70
    assert st._gap_cm(0.64) == 60
