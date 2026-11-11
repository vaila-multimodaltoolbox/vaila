"""The goalkeeper's penalty dive (vaila/freekiki_keeper.py) on a synthetic keeper with known motion.

A rigid keeper (all keypoints move together) shifts sideways with a constant acceleration from a
known onset and steps forward, so his centre of mass, its speeds, the side, the step and the angle
are known in closed form. Law 14 is checked from the 3D feet and from pixels through a known ground
homography. Synthetic data only.

Version: 0.4.16
Update Date: 11 November 2026
"""

from __future__ import annotations

import numpy as np
import pytest

from vaila import freekiki_keeper as kp

FPS = 30.0
N = 40
GOAL_X = 52.45
ONSET, CONTACT, SAVE = 12, 17, 28
ACCEL = 6.0  # m/s^2 toward -y (the keeper's left, facing the kicker of the right goal)
FORWARD = 0.25  # m: the step toward the kicker, done by the onset + 6 frames

# the body at rest: hips 1.0 m high, 0.45 m in front of the goal line
BODY = {
    "left-ear": (-0.05, -0.08, 1.75), "right-ear": (-0.05, 0.08, 1.75),
    "left-shoulder": (0.0, -0.2, 1.45), "right-shoulder": (0.0, 0.2, 1.45),
    "left-elbow": (-0.1, -0.35, 1.2), "right-elbow": (-0.1, 0.35, 1.2),
    "left-wrist": (-0.2, -0.4, 1.0), "right-wrist": (-0.2, 0.4, 1.0),
    "left-index-tip": (-0.25, -0.42, 0.95), "right-index-tip": (-0.25, 0.42, 0.95),
    "left-pinky-tip": (-0.25, -0.38, 0.95), "right-pinky-tip": (-0.25, 0.38, 0.95),
    "left-hip": (0.0, -0.12, 1.0), "right-hip": (0.0, 0.12, 1.0),
    "left-knee": (-0.05, -0.15, 0.55), "right-knee": (-0.05, 0.15, 0.55),
    "left-ankle": (0.0, -0.2, 0.1), "right-ankle": (0.0, 0.2, 0.1),
    "left-heel": (0.06, -0.2, 0.03), "right-heel": (0.06, 0.2, 0.03),
    "left-big-toe-tip": (-0.18, -0.22, 0.02), "right-big-toe-tip": (-0.18, 0.22, 0.02),
}  # fmt: skip
START = np.array([GOAL_X - 0.45, 0.0, 0.0])


def _shift(f: int) -> np.ndarray:
    """The rigid body's displacement at frame ``f``."""
    t = max(f - ONSET, 0) / FPS
    forward = -FORWARD * min(
        max(f - ONSET, 0) / 6.0, 1.0
    )  # toward the kicker (-x for the right goal)
    return np.array([forward, -0.5 * ACCEL * t * t, 0.0])


def _points() -> dict[str, np.ndarray]:
    return {
        name: np.array([START + np.asarray(off) + _shift(f) for f in range(N)])
        for name, off in BODY.items()
    }


def test_the_dive_side_displacement_and_speeds_are_recovered() -> None:
    result = kp.keeper_kinematics(_points(), fps=FPS, contact_frame=CONTACT, save_frame=SAVE,
                                  goal_x=GOAL_X, kick_time_s=(CONTACT - 0.5) / FPS)  # fmt: skip
    assert result["available"]
    m = result["metrics"]
    assert m["dive_side"] == "left"  # -y is the keeper's left, facing the kicker of the right goal
    # the first frame whose lateral speed (a t) passes 0.5 m/s; the onset needs 3 such frames
    speeds = [ACCEL * max(f - ONSET, 0) / FPS for f in range(N)]
    first = next(f for f in range(N) if speeds[f] > kp.ONSET_SPEED_MPS)
    assert abs(m["onset_frame_auto"] - first) <= 1  # the local fit smooths the corner by a frame
    assert m["onset_source"] == "com_lateral_speed"
    onset = m["onset_frame"]
    t = (SAVE - ONSET) / FPS
    total = 0.5 * ACCEL * t * t
    t0 = max(onset - ONSET, 0) / FPS
    assert m["lateral_displacement_m"] == pytest.approx(total - 0.5 * ACCEL * t0 * t0, abs=0.01)
    assert m["peak_lateral_speed_mps"] == pytest.approx(ACCEL * t, rel=0.05)
    assert m["frontal_step_m"] == pytest.approx(
        FORWARD - (FORWARD * min(t0 * FPS / 6.0, 1.0)), abs=0.02
    )
    assert m["departure_angle_deg"] is not None and 0.0 < m["departure_angle_deg"] < 45.0


def test_an_onset_marked_by_eye_is_used_and_compared() -> None:
    result = kp.keeper_kinematics(_points(), fps=FPS, contact_frame=CONTACT, save_frame=SAVE,
                                  goal_x=GOAL_X, onset_frame_given=ONSET)  # fmt: skip
    m = result["metrics"]
    assert (m["onset_frame"], m["onset_source"]) == (ONSET, "given")
    assert m["onset_ms_before_kick"] == pytest.approx((CONTACT - ONSET) / FPS * 1000.0, abs=0.1)
    assert m["onset_frame_auto"] is not None and m["onset_frame_auto"] > ONSET


def test_the_hands_ball_gap_and_law14_from_the_3d_feet() -> None:
    points = _points()
    ball = np.full((N, 3), np.nan)
    hands = kp.palms(points, N)
    ball[SAVE] = hands[SAVE] + np.array([-(0.11 + 0.05), 0.0, 0.0])  # the ball 5 cm off the palms
    result = kp.keeper_kinematics(points, fps=FPS, contact_frame=CONTACT, save_frame=SAVE,
                                  goal_x=GOAL_X, ball_xyz=ball)  # fmt: skip
    m = result["metrics"]
    assert m["hands_ball_gap_m"] == pytest.approx(0.05, abs=1e-6)
    feet = m["law14_feet_m"][CONTACT]
    heel_x = (START + np.asarray(BODY["left-heel"]) + _shift(CONTACT))[0]
    assert feet["left-heel"] == pytest.approx(GOAL_X - heel_x, abs=1e-3)
    assert m["law14_nearest_m"][CONTACT] == pytest.approx(min(feet.values()))
    assert kp.law14_status(m["law14_nearest_m"][CONTACT]) == "in_front"


def test_law14_status_limits() -> None:
    assert kp.law14_status(-0.05) == "on_or_behind_line"
    assert kp.law14_status(0.06) == "touching_the_line"
    assert kp.law14_status(0.40) == "in_front"
    assert kp.law14_status(0.10, sigma_m=0.05) == "undecided"  # the interval crosses the line edge


def _ground_homography() -> np.ndarray:
    """8 DLT2D coefficients of a camera behind the goal line looking at it (ground -> pixel)."""
    c = np.array([60.0, -8.0, 12.0])
    target = np.array([GOAL_X, 0.0, 0.0])
    fwd = (target - c) / np.linalg.norm(target - c)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.vstack([right, down, fwd])
    K = np.array([[2000.0, 0.0, 960.0], [0.0, 2000.0, 540.0], [0.0, 0.0, 1.0]])
    P = K @ np.hstack([R, (-R @ c)[:, None]])
    H = P[:, [0, 1, 3]] / P[2, 3]
    return np.array([H[0, 0], H[0, 1], H[0, 2], H[1, 0], H[1, 1], H[1, 2], H[2, 0], H[2, 1]])


def _pixel(L: np.ndarray, x: float, y: float) -> tuple[float, float]:
    H = np.array([[L[0], L[1], L[2]], [L[3], L[4], L[5]], [L[6], L[7], 1.0]])
    h = H @ np.array([x, y, 1.0])
    return float(h[0] / h[2]), float(h[1] / h[2])


def test_law14_from_pixels_measures_the_foot_against_the_painted_line() -> None:
    L = _ground_homography()
    foot = _pixel(L, GOAL_X - 0.30, -0.6)  # the heel 0.30 m in front of the line
    line = [_pixel(L, GOAL_X, y) for y in (-1.5, -0.5, 0.5)]
    out = kp.law14_from_pixels(foot, line, L, GOAL_X, sigma_px=2.0)
    assert out["in_front_m"] == pytest.approx(0.30, abs=1e-3)
    assert out["line_offset_from_model_m"] == pytest.approx(0.0, abs=1e-6)
    assert 0.0 < out["sigma_m"] < 0.2 and out["status"] == "in_front"
    # a calibration offset common to the foot and the line cancels out
    shifted = [(u, v + 4.0) for u, v in line]
    moved = kp.law14_from_pixels((foot[0], foot[1] + 4.0), shifted, L, GOAL_X)
    assert moved["in_front_m"] == pytest.approx(0.30, abs=0.03)
    on_line = kp.law14_from_pixels(_pixel(L, GOAL_X + 0.05, 0.0), line, L, GOAL_X, sigma_px=0.5)
    assert on_line["in_front_m"] == pytest.approx(-0.05, abs=1e-3)
    assert on_line["status"] == "on_or_behind_line"


def test_rear_override_replaces_the_monocular_lateral_travel(tmp_path) -> None:
    """A rear camera's lateral travel replaces the one-camera value and drops what came from it."""
    keeper = {
        "lateral_displacement_m": 0.357,
        "movement_time_s": 0.5339,
        "peak_lateral_speed_mps": 2.327,
        "peak_resultant_speed_mps": 2.7,
        "hands_ball_gap_m": 0.105,
        "frontal_step_m": 0.461,
        "frontal_step_foot": "left",
        "com_drop_m": 0.425,
        "dive_side": "left",
    }
    override = {
        "lateral_displacement_m": 0.94,
        "lateral_range_m": [0.75, 1.14],
        "hip_forward_m": 0.63,
    }
    out = kp.apply_rear_override(keeper, override)
    assert out["lateral_displacement_m"] == pytest.approx(0.94)
    assert out["lateral_displacement_model_1cam_m"] == pytest.approx(0.357)
    assert out["lateral_displacement_range_m"] == [0.75, 1.14]
    assert out["lateral_source"] == "rear_camera"
    assert out["mean_lateral_speed_mps"] == pytest.approx(0.94 / 0.5339, abs=0.01)
    assert out["hip_forward_m"] == pytest.approx(0.63)
    for key in kp.REAR_NULLED_KEYS:
        if key in keeper:
            assert out[key] is None
    # what the poses measure well is kept, and the input is not modified
    assert out["com_drop_m"] == pytest.approx(0.425) and out["dive_side"] == "left"
    assert keeper["lateral_displacement_m"] == pytest.approx(0.357)
    # no override: the same keeper comes back
    assert kp.apply_rear_override(keeper, None) is keeper

    import json

    (tmp_path / kp.REAR_OVERRIDE_FILE).write_text(json.dumps(override), encoding="utf-8")
    metrics = {"keeper": keeper, "interception": {"in_front_of_line_m": 1.16}}
    patched = kp.apply_rear_to_metrics(metrics, tmp_path)
    assert (
        patched["keeper"]["lateral_source"] == "rear_camera"
        and patched["rear_override"] == override
    )
    assert metrics["keeper"]["lateral_displacement_m"] == pytest.approx(0.357)
    assert kp.apply_rear_to_metrics(metrics, tmp_path / "missing") is metrics
