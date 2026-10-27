"""Tests for the physically consistent tracks of the ball and the players (no GPU, no video)."""

from __future__ import annotations

import numpy as np
import pytest

from vaila import freekiki_physics as ph

FPS = 30.0


def _speeds_and_accels(points: dict[int, tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    frames = sorted(points)
    p = np.array([points[f] for f in frames], dtype=float)
    dt = np.diff(frames) / FPS
    v = np.linalg.norm(np.diff(p, axis=0), axis=1) / dt
    a = np.abs(np.diff(v)) / dt[1:]
    return v, a


def test_a_wrong_cluster_of_points_is_rejected() -> None:
    # A straight run at 5 m/s, with nine frames in a row placed 5 m off the path.
    truth = {f: (-40.0 + f * 5.0 / FPS, 9.0) for f in range(90)}
    raw = dict(truth)
    for f in range(40, 49):
        raw[f] = (truth[f][0], truth[f][1] + 5.0)
    out = ph.smooth_track(raw, 90, FPS, ph.PLAYER_MODEL)
    for f in range(40, 49):
        assert np.hypot(out[f][0] - truth[f][0], out[f][1] - truth[f][1]) < 0.5
    v, a = _speeds_and_accels(out)
    assert v.max() < 7.0 and a.max() < 10.0


def test_a_moving_track_keeps_its_speed() -> None:
    truth = {f: (f * 6.0 / FPS, 0.0) for f in range(60)}
    out = ph.smooth_track(truth, 60, FPS, ph.PLAYER_MODEL)
    frames = sorted(out)
    assert frames == list(range(60))
    assert np.allclose([out[f][0] for f in frames], [truth[f][0] for f in frames], atol=0.05)


def test_the_ball_speed_is_capped_at_the_physical_ceiling() -> None:
    # Measurements that move 3 m per frame (90 m/s): the track must not exceed 40 m/s.
    raw = {f: (f * 3.0, 0.0) for f in range(20)}
    out = ph.smooth_track(raw, 20, FPS, ph.BALL_MODEL)
    v, _ = _speeds_and_accels(out)
    assert v.max() <= ph.BALL_MODEL.max_speed_mps + 1e-6


def test_frames_far_from_any_measurement_stay_empty() -> None:
    raw = {0: (0.0, 0.0), 1: (0.01, 0.0), 40: (5.0, 0.0), 41: (5.01, 0.0)}
    out = ph.smooth_track(raw, 50, FPS, ph.BALL_MODEL)  # ball model: a gap of 6 frames
    assert {0, 1, 40, 41} <= set(out)
    assert 20 not in out  # nothing invented across a long gap
    assert all(f in out for f in range(0, 2))


def test_a_track_that_really_moved_restarts_after_repeated_rejections() -> None:
    raw = dict.fromkeys(range(10), (0.0, 0.0))
    raw.update(dict.fromkeys(range(10, 30), (30.0, 0.0)))  # a new position for good
    out = ph.smooth_track(raw, 30, FPS, ph.PLAYER_MODEL)
    assert abs(out[29][0] - 30.0) < 0.5


def test_empty_input_gives_no_track() -> None:
    assert ph.smooth_track({}, 10, FPS, ph.PLAYER_MODEL) == {}


def _look_at_projection(f: float, centre, target, width: float = 1920.0, height: float = 1080.0):
    """3x4 projection (pitch metres -> pixels) of a camera at ``centre`` looking at ``target``."""
    centre = np.asarray(centre, dtype=float)
    target = np.asarray(target, dtype=float)
    up = np.array([0.0, 0.0, 1.0])
    z = target - centre
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    t = -R @ centre
    K = np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])
    return K @ np.hstack([R, t[:, None]])


def test_flight_without_drag_or_spin_is_the_projectile_solution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vaila import freekiki_physics as ph

    monkeypatch.setattr(ph, "drag_coefficient", lambda v, sp: 0.0)
    monkeypatch.setattr(ph, "lift_coefficient", lambda sp: 0.0)
    theta = np.array([0.0, 0.0, 27.8, -0.4, 8.8, 0.0, 0.0, 0.0, 0.0])
    t = np.array([0.0, 1.0, 3.2])
    sim = ph.simulate_flight(theta, t, rest_height=0.0)
    exact = np.column_stack([27.8 * t, -0.4 * t, 8.8 * t - 0.5 * 9.81 * t * t])
    assert np.abs(sim - exact).max() < 1e-6


def test_drag_coefficient_follows_the_brazuca_fit() -> None:
    from vaila import freekiki_physics as ph

    subcritical = ph.drag_coefficient(10.0, 0.0)  # Asai & Seo: about 0.47 below the crisis
    supercritical = ph.drag_coefficient(30.0, 0.0)  # 0.12 to 0.15 above it (Brazuca fit ~0.2)
    assert 0.40 < subcritical < 0.52
    assert 0.15 < supercritical < 0.25
    assert ph.drag_coefficient(30.0, 0.2) > supercritical  # the spin term adds drag
    assert ph.drag_coefficient(60.0, 0.0) == pytest.approx(ph.drag_coefficient(35.0, 0.0))


def test_lift_grows_with_spin_and_vanishes_without_it() -> None:
    from vaila import freekiki_physics as ph

    assert ph.lift_coefficient(0.0) == 0.0
    assert 0.0 < ph.lift_coefficient(0.1) < ph.lift_coefficient(0.2)


def test_magnus_spin_bends_the_flight() -> None:
    from vaila import freekiki_physics as ph

    t = np.array([0.0, 1.5])
    base = np.array([0.0, 0.0, 20.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0])  # x0,y0,vx,vy,vz,tk,w
    spun = base.copy()
    spun[6:9] = [0.0, 0.0, -2 * np.pi * 5.0]
    straight = ph.simulate_flight(base, t, rest_height=0.0)[-1]
    curved = ph.simulate_flight(spun, t, rest_height=0.0)[-1]
    assert np.linalg.norm(curved - straight) > 0.5  # metres of Magnus deflection


def test_fit_flight_recovers_a_spinning_flight_from_noisy_pixels() -> None:
    from vaila import freekiki_physics as ph

    fps, n = 30.0, 31
    theta = np.array([-30.0, -20.0, 22.0, 14.0, 7.0, 0.1, 0.0, 0.0, -2 * np.pi * 5.0])
    times = np.arange(n) / fps
    truth = ph.simulate_flight(theta, times, rest_height=ph.BALL_RADIUS)
    P = _look_at_projection(2400.0, [-60.0, -45.0, 25.0], [-30.0, -10.0, 0.0])
    cams = dict.fromkeys(range(n), P)
    rng = np.random.default_rng(3)
    pixels = {}
    for f in range(n):
        h = P @ np.append(truth[f], 1.0)
        pixels[f] = (h[0] / h[2] + rng.normal(0, 0.5), h[1] / h[2] + rng.normal(0, 0.5))
    anchors = {
        "rest": (-30.0, -20.0),
        "barrier": {
            "frame": 9,
            "xy": (truth[9, 0] + 0.3, truth[9, 1]),
            "z": truth[9, 2],
            "offset": 0.3,
        },
        "keeper": {"frame": 30, "xyz": tuple(truth[30])},
    }
    fit = ph.fit_flight(pixels, cams, fps, n, anchors=anchors)
    assert fit["success"]
    err = np.array([np.linalg.norm(np.subtract(fit["positions"][f], truth[f])) for f in range(n)])
    assert err.max() < 0.25, err.max()
    assert fit["rms_px"] < 1.5
    assert fit["spin_rps"] == pytest.approx(5.0, rel=0.1)


def test_fit_flight_with_three_point_anchors_recovers_the_flight() -> None:
    from vaila import freekiki_physics as ph

    fps, n = 30.0, 31
    theta = np.array([-30.0, -20.0, 22.0, 14.0, 7.0, 0.1, 0.0, 0.0, -2 * np.pi * 5.0])
    times = np.arange(n) / fps
    truth = ph.simulate_flight(theta, times, rest_height=ph.BALL_RADIUS)
    P = _look_at_projection(2400.0, [-60.0, -45.0, 25.0], [-30.0, -10.0, 0.0])
    cams = dict.fromkeys(range(n), P)
    rng = np.random.default_rng(5)
    pixels = {}
    for f in range(n):
        h = P @ np.append(truth[f], 1.0)
        pixels[f] = (h[0] / h[2] + rng.normal(0, 0.5), h[1] / h[2] + rng.normal(0, 0.5))
    # the three anchors a user picks: the start on the ground, a point in the air, the goal
    points = [
        {"frame": 0, "xyz": [-30.0, -20.0, ph.BALL_RADIUS], "sigma": 0.2, "label": "início"},
        {"frame": 12, "xyz": list(truth[12]), "sigma": 0.8, "label": "meio"},
        {"frame": 30, "xyz": list(truth[30]), "sigma": 0.2, "label": "chegada"},
    ]
    fit = ph.fit_flight(pixels, cams, fps, n, anchors={"points": points})
    assert fit["success"]
    err = np.array([np.linalg.norm(np.subtract(fit["positions"][f], truth[f])) for f in range(n)])
    assert err.max() < 0.3, err.max()
    assert set(fit["anchor_residuals"]) >= {"p0_m", "p1_m", "p2_m"}
