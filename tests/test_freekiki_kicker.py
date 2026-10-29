"""The kicker's metrics (vaila/freekiki_kicker.py) on a synthetic kick with known values.

The scene is built so that every number has a closed form: the foot moves at 10 m/s, the ball
leaves at 30 m/s, the hips move 3 m/s across the approach, and the ball crosses the goal plane 1 m
to the side and 0.5 m above the centre of the goal mouth. A metric with no data must come back
empty with the reason, never with a number.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from vaila import freekiki_kicker as kk

FPS = 60.0
N = 120
CONTACT = 40  # the ball starts to move at frame 40
KICKER = 1


def _ball() -> np.ndarray:
    f = np.arange(N)
    x = 10.0 + 30.0 * f / FPS  # 30 m/s towards the goal (x = +52.45)
    return np.column_stack([x, np.full(N, 1.0), np.full(N, 1.72)])


def _pixels() -> dict[int, tuple[float, float]]:
    """Static until the contact, then 50 px away: the onset is the contact frame."""
    return {f: (500.0 + (50.0 if f >= CONTACT else 0.0), 500.0) for f in range(N)}


def _rows() -> list[dict]:
    ball = _ball()
    f = np.arange(N)
    foot = np.column_stack([ball[CONTACT, 0] + 10.0 * (f - CONTACT) / FPS, np.full(N, 1.0),
                            np.full(N, 1.72)])  # fmt: skip
    support = np.tile([ball[CONTACT, 0], 0.7, 0.11], (N, 1))
    hip = np.column_stack([np.full(N, 5.0), 3.0 * f / FPS, np.full(N, 1.0)])
    knee = hip + np.array([0.0, 0.0, -0.5])
    ankle = knee + np.array([0.5, 0.0, 0.0])
    points = {
        "left-big-toe-tip": foot, "left-small-toe-tip": foot, "left-heel": foot,
        "right-ankle": support, "right-heel": support, "right-big-toe-tip": support,
        "right-small-toe-tip": support, "left-hip": hip, "right-hip": hip,
        "left-knee": knee, "left-ankle": ankle,
    }  # fmt: skip
    rows = []
    for name, arr in points.items():
        for frame in range(N):
            rows.append({
                "frame": frame, "player_id": KICKER, "kpt_idx": 0, "kpt_name": name,
                "x_m": float(arr[frame, 0]), "y_m": float(arr[frame, 1]), "z_m": float(arr[frame, 2]),
                "valid": True,
            })  # fmt: skip
    return rows


def _metric(result: dict, name: str):
    return next(m for m in result["metrics"] if m["metric"] == name)


def test_local_velocity_is_exact_for_linear_and_quadratic_motion() -> None:
    t = np.arange(20) / FPS
    lin = kk.local_velocity(10.0 * t, FPS)
    assert np.allclose(lin[3:-3], 10.0)
    quad = kk.local_velocity(0.5 * 4.0 * t**2, FPS)
    assert np.allclose(quad[3:-3], 4.0 * t[3:-3], atol=1e-9)


def test_local_velocity_is_missing_where_the_window_has_too_few_frames() -> None:
    values = np.array([np.nan, np.nan, 1.0, 2.0, np.nan])
    out = kk.local_velocity(values, FPS)
    assert np.isnan(out[0]) and np.isnan(out[1])


def test_angle_between_a_right_angle_and_missing_vectors() -> None:
    assert math.isclose(float(kk.angle_between([1.0, 0.0], [0.0, 2.0])), 90.0, abs_tol=1e-9)
    assert np.isnan(kk.angle_between([0.0, 0.0], [1.0, 0.0]))


def test_proximal_distal_order_needs_all_three_peaks() -> None:
    assert kk.proximal_distal(1, 2, 3) is True
    assert kk.proximal_distal(3, 2, 1) is False
    assert kk.proximal_distal(None, 2, 3) is None


def test_the_kick_metrics_match_the_scene() -> None:
    result = kk.analyze_kicker(
        _rows(), _ball(), _pixels(), kicker_id=KICKER, fps=FPS, kick_time_s=CONTACT / FPS
    )
    assert _metric(result, "contact_frame")["value"] == CONTACT
    assert _metric(result, "kick_foot")["value"] == "left"
    assert math.isclose(_metric(result, "v_foot_pre_mps")["value"], 10.0, abs_tol=1e-6)
    assert math.isclose(_metric(result, "v_ball_post_mps")["value"], 30.0, abs_tol=1e-6)
    assert math.isclose(_metric(result, "ratio_vball_vfoot")["value"], 3.0, abs_tol=1e-6)
    support = np.array([_ball()[CONTACT, 0], 0.7, 0.11])
    expected = float(np.linalg.norm(support - _ball()[CONTACT]))
    assert math.isclose(_metric(result, "d_support_ball_m")["value"], expected, abs_tol=1e-6)
    assert math.isclose(_metric(result, "approach_angle_deg")["value"], 90.0, abs_tol=1e-3)
    assert math.isclose(_metric(result, "knee_angle_contact_deg")["value"], 90.0, abs_tol=1e-6)
    assert abs(_metric(result, "knee_ext_peak_deg_s")["value"]) < 1e-6
    assert math.isclose(
        _metric(result, "goal_error_m")["value"], math.hypot(1.0, 0.5), abs_tol=1e-3
    )
    assert math.isclose(_metric(result, "goal_lateral_m")["value"], 1.0, abs_tol=1e-3)
    assert math.isclose(_metric(result, "goal_height_m")["value"], 1.72, abs_tol=1e-3)
    assert abs(_metric(result, "kick_time_check_s")["value"]) < 1e-9
    assert result["warnings"] == []


def test_the_sources_and_the_sensitivity_are_reported() -> None:
    result = kk.analyze_kicker(_rows(), _ball(), _pixels(), kicker_id=KICKER, fps=FPS)
    assert _metric(result, "v_foot_pre_mps")["source"] == "filtrado"
    assert _metric(result, "v_ball_post_mps")["source"] == "ajustado"
    assert _metric(result, "goal_error_m")["source"] == "ajustado"
    assert _metric(result, "approach_angle_deg")["source"] == "inferido"
    assert len(result["sensitivity"]) == len(kk.SENSITIVITY_S)
    # 17 ms is one frame at 60 Hz, 67 ms four frames: the speeds depend on the window
    assert [row["frames"] for row in result["sensitivity"]] == [1, 2, 4]


def test_the_foot_can_be_given_by_the_user() -> None:
    result = kk.analyze_kicker(
        _rows(), _ball(), _pixels(), kicker_id=KICKER, fps=FPS, kick_foot="right"
    )
    assert _metric(result, "kick_foot")["value"] == "right"
    assert _metric(result, "kick_foot")["reason"] == "given by the user"


def test_a_missing_goal_crossing_is_empty_with_the_reason() -> None:
    ball = _ball()
    ball[CONTACT:, 0] = ball[CONTACT, 0]  # the ball stops at the contact: a miss or a save
    result = kk.analyze_kicker(_rows(), ball, _pixels(), kicker_id=KICKER, fps=FPS)
    goal = _metric(result, "goal_error_m")
    assert goal["value"] == "" and goal["quality"] == "missing"
    assert "no crossing" in goal["reason"]


def test_a_kicker_without_points_gives_no_numbers() -> None:
    result = kk.analyze_kicker([], _ball(), _pixels(), kicker_id=KICKER, fps=FPS)
    assert result["series"] is None
    contact = _metric(result, "contact_frame")
    assert contact["value"] == "" and contact["quality"] == "missing"
    assert contact["reason"] == "kicker has no points"
    assert result["warnings"]


def test_no_ball_movement_means_no_contact() -> None:
    still = dict.fromkeys(range(N), (500.0, 500.0))
    result = kk.analyze_kicker(_rows(), _ball(), still, kicker_id=KICKER, fps=FPS)
    contact = _metric(result, "contact_frame")
    assert contact["value"] == "" and contact["quality"] == "missing"


def test_write_outputs_writes_the_files(tmp_path: Path) -> None:
    result = kk.analyze_kicker(_rows(), _ball(), _pixels(), kicker_id=KICKER, fps=FPS)
    paths = kk.write_outputs(result, tmp_path / "kicker")
    for key in ("metrics", "json", "series", "sensitivity", "figure"):
        assert paths[key].is_file() and paths[key].stat().st_size > 0
    header = paths["metrics"].read_text(encoding="utf-8").splitlines()[0]
    assert header == "metric,value,unit,source,quality,reason,definition"
