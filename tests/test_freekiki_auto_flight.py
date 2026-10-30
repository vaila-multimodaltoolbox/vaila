"""Monocular ball flight with automatic start (kick contact) and arrival (goal entry or keeper's hand).

A physical flight is simulated, seen by one synthetic camera, and fitted back; the entry point from the
goal homography is compared with the true crossing. Synthetic data only: no GPU, no model, no video.
"""

from __future__ import annotations

import numpy as np
import pytest

from vaila import freekiki_auto_flight as af
from vaila.fifa_to_dlt import compute_dlt3d_from_KRt
from vaila.freekiki_field3d import ball_3d
from vaila.freekiki_physics import BALL_RADIUS, seed_flight, simulate_flight
from vaila.monocular_dlt_align import dlt_projection_matrix

W, H = 1920, 1080
FPS = 30.0
N = 60
K = np.array([[2400.0, 0.0, W / 2.0], [0.0, 2400.0, H / 2.0], [0.0, 0.0, 1.0]])


def _camera(centre, target) -> np.ndarray:
    centre, target = np.asarray(centre, float), np.asarray(target, float)
    z = target - centre
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    return compute_dlt3d_from_KRt(K, R, -R @ centre)


L = _camera([10.0, -38.0, 22.0], [50.0, 0.0, 1.0])


def _pix(xyz) -> np.ndarray:
    h = dlt_projection_matrix(L) @ np.append(np.asarray(xyz, float), 1.0)
    return h[:2] / h[2]


def _flight(theta) -> np.ndarray:
    return simulate_flight(np.asarray(theta, float), np.arange(N) / FPS, BALL_RADIUS)


def _pixels(positions: np.ndarray) -> dict[int, tuple[float, float]]:
    return {f: tuple(_pix(positions[f])) for f in range(N)}


def _ball_rows(pixels: dict[int, tuple[float, float]]) -> list[dict]:
    return [
        {"frame": f, "cx": uv[0], "cy": uv[1], "score": 1.0} for f, uv in sorted(pixels.items())
    ]


def _truth_crossing(positions: np.ndarray, x_goal: float) -> tuple[float, np.ndarray]:
    """The fractional frame and the (y, z) where the true track reaches the goal plane."""
    for f in range(1, N):
        if positions[f - 1, 0] < x_goal <= positions[f, 0]:
            w = (x_goal - positions[f - 1, 0]) / (positions[f, 0] - positions[f - 1, 0])
            p = positions[f - 1] + w * (positions[f] - positions[f - 1])
            return f - 1 + w, p[1:]
    raise AssertionError("the true track does not reach the goal plane")


def test_kick_onset_is_the_first_frame_that_leaves_the_rest_pixel() -> None:
    still = dict.fromkeys(range(20), (100.0, 100.0))
    moving = {f: (100.0 + (f - 10) * 3.0 if f >= 10 else 100.0, 100.0) for f in range(20)}
    assert af.kick_onset(still) is None
    assert af.kick_onset(moving) == 12  # 3 px per frame from frame 10: more than 5 px at frame 12


def test_goal_plane_point_recovers_a_point_on_the_plane() -> None:
    truth = np.array([af.GOAL_X_M, 1.0, 1.2])
    yz = af.goal_plane_point(dlt_projection_matrix(L), _pix(truth), af.GOAL_X_M)
    assert yz == pytest.approx([1.0, 1.2], abs=1e-6)
    assert af.inside_goal_mouth(yz)
    assert not af.inside_goal_mouth(np.array([5.0, 1.0]))  # outside the posts


def test_seed_flight_reads_rest_point_and_launch() -> None:
    frames = np.arange(20, dtype=float)
    xyz = np.zeros((20, 3))
    xyz[:, 0] = 30.0
    xyz[:, 2] = 0.11  # on the ground until the launch
    xyz[6:, 2] = 0.5 + 2.0 * (frames[6:] - 6) / FPS  # above launch_height_m (0.3) from frame 6
    theta = seed_flight(frames, xyz, FPS, start_frame=0)
    assert theta[0] == pytest.approx(30.0) and theta[1] == pytest.approx(0.0)
    assert theta[4] == pytest.approx(2.0)  # vz between frames 6 and 9
    assert theta[5] == pytest.approx(5 / FPS, abs=1e-9)  # one frame before the launch at frame 6


def test_entry_is_found_and_the_fit_follows_the_true_flight() -> None:
    theta = [
        30.0,
        0.0,
        22.0,
        0.0,
        7.0,
        0.4,
        0.0,
        0.0,
        0.0,
    ]  # a kick to the goal, lands in the mouth
    positions = _flight(theta)
    pixels = _pixels(positions)
    cameras = dict.fromkeys(range(N), L)
    ball_all = ball_3d(_ball_rows(pixels), cameras)
    result = af.auto_flight(ball_all, pixels, cameras, [], FPS, N)
    assert result is not None
    assert (
        result["onset_frame"] == 13
    )  # the launch is at 0.4 s (frame 12); frame 13 is the first to move 5 px
    arrival = result["arrival"]
    assert arrival is not None and arrival["type"] == "entry"
    frame_true, yz_true = _truth_crossing(positions, af.GOAL_X_M)
    assert abs(arrival["frame"] - frame_true) <= 1.5
    assert np.allclose(arrival["xyz"][1:], yz_true, atol=0.1)  # the homography entry point
    kick_t = result["fit"]["theta"][5]  # the flight starts at the contact, not before it
    assert (result["onset_frame"] - 1) / FPS - 1e-6 <= kick_t <= result["onset_frame"] / FPS + 1e-6
    fitted = result["fit"]["positions"]
    errors = [np.linalg.norm(np.asarray(fitted[f]) - positions[f]) for f in range(12, 44)]
    assert float(np.median(errors)) < 0.15
    assert result["fit"]["rms_px"] < 1.0


def test_a_save_takes_the_keeper_hand_at_the_last_frame() -> None:
    theta = [
        30.0,
        0.0,
        9.0,
        0.0,
        6.0,
        0.4,
        0.0,
        0.0,
        0.0,
    ]  # short kick: the ball never reaches the goal
    positions = _flight(theta)
    pixels = _pixels(positions)
    cameras = dict.fromkeys(range(N), L)
    ball_all = ball_3d(_ball_rows(pixels), cameras)
    pose_rows = []
    last = N - 1
    for name, xyz in (
        ("left-ankle", (51.2, 0.0, 0.0)),
        ("right-ankle", (51.4, 0.2, 0.0)),
        ("left-wrist", (51.9, -0.3, 1.1)),
        ("right-wrist", (51.7, -0.1, 1.2)),
    ):
        pose_rows.append(
            {
                "frame": last,
                "person_id": 1,
                "kpt_idx": 0,
                "kpt_name": name,
                "x_m": xyz[0],
                "y_m": xyz[1],
                "z_m": xyz[2],
            }  # fmt: skip
        )
    result = af.auto_flight(ball_all, pixels, cameras, pose_rows, FPS, N)
    assert result is not None
    arrival = result["arrival"]
    assert arrival is not None and arrival["type"] == "keeper_hand"
    assert np.allclose(arrival["xyz"], [51.8, -0.2, 1.15], atol=1e-6)  # mean of the two wrists


def test_no_movement_means_no_automatic_flight() -> None:
    still = {f: tuple(_pix([30.0, 0.0, BALL_RADIUS])) for f in range(N)}
    cameras = dict.fromkeys(range(N), L)
    ball_all = ball_3d(_ball_rows(still), cameras)
    assert af.auto_flight(ball_all, still, cameras, [], FPS, N) is None


def test_a_drifting_marker_before_the_contact_does_not_fix_the_kick_time() -> None:
    # the hand-marked ball slides a few pixels before the real contact (frame 12), so the first frame
    # that moves 5 px is early (frame 8): the flight must still start at the contact, not at the drift
    theta = [30.0, 0.0, 22.0, 0.0, 7.0, 0.4, 0.0, 0.0, 0.0]
    positions = _flight(theta)
    pixels = _pixels(positions)
    for f in range(4, 12):
        u, v = pixels[f]
        pixels[f] = (u + (f - 3) * 1.0, v)
    cameras = dict.fromkeys(range(N), L)
    ball_all = ball_3d(_ball_rows(pixels), cameras)
    result = af.auto_flight(ball_all, pixels, cameras, [], FPS, N)
    assert result is not None
    assert result["onset_frame"] < 12  # the drift makes the first moved frame early
    assert result["fit"]["kick_time_s"] == pytest.approx(0.4, abs=1.5 / FPS)
    assert result["fit"]["rms_px"] < 3.0
