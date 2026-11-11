"""Monocular ball flight with automatic start (kick contact) and arrival (goal entry or keeper's hand).

A physical flight is simulated, seen by one synthetic camera, and fitted back; the entry point from the
goal homography is compared with the true crossing. Synthetic data only: no GPU, no model, no video.

Version: 0.4.15
Update Date: 09 November 2026
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


def test_a_missing_entry_is_undetermined_and_a_save_needs_explicit_confirmation() -> None:
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
    assert result["arrival"] is None
    _, record = af.auto_rows_and_record(result, pixels, N)
    assert record["arrival_status"] == "undetermined"
    assert "p_arrival" not in record["anchor_points"]
    observation = {
        "type": "keeper_hand",
        "frame": last,
        "xyz": [51.8, -0.2, 1.15],
        "confirmed": True,
        "sigma_m": 0.3,
    }
    result = af.auto_flight(
        ball_all, pixels, cameras, pose_rows, FPS, N, arrival_observation=observation
    )
    arrival = result["arrival"]
    assert arrival is not None and arrival["type"] == "keeper_hand"
    assert np.allclose(arrival["xyz"], [51.8, -0.2, 1.15], atol=1e-6)  # mean of the two wrists
    assert arrival["observation"]["confirmed"] is True


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


KICK_POINT = (30.0, 0.0)
SHOT = [
    KICK_POINT[0],
    KICK_POINT[1],
    22.0,
    0.0,
    7.0,
    0.4,
    0.0,
    0.0,
    0.0,
]  # kicked at 0.4 s (frame 12)
ROLL = np.array([0.0, 6.0])  # a cross: the ball rolls into the kick at 6 m/s along +y


def _rolling_positions() -> np.ndarray:
    """The ball rolls on the ground into the kick point, then flies (the kick at frame 12)."""
    positions = _flight(SHOT)
    kick_t = SHOT[5]
    for f in range(N):
        t = f / FPS
        if t < kick_t:
            xy = np.asarray(KICK_POINT) + ROLL * (t - kick_t)
            positions[f] = [xy[0], xy[1], BALL_RADIUS]
    return positions


def test_rolling_break_finds_the_kick_despite_a_badly_marked_frame() -> None:
    ground = {f: (5.0 + 0.2 * f, 1.0) for f in range(10)}  # 6 m/s at 30 fps
    ground.update({f: (7.0 + 0.8 * (f - 10), 1.0 + 0.3 * (f - 10)) for f in range(10, 20)})
    ground[4] = (ground[4][0] + 0.15, ground[4][1] - 0.15)  # one bad mark while it rolls
    found = af.rolling_break(ground, FPS)
    # frame 10 is the kick point itself, still on the rolling line: 11 is the first frame off it
    assert found is not None and found["frame"] == 11
    assert found["kick_xy"] == pytest.approx([7.1, 1.0], abs=0.05)  # between frames 10 and 11
    assert found["incoming"]["speed_mps"] == pytest.approx(6.0, abs=0.5)


def test_a_ball_flying_from_the_first_frame_has_no_rolling_break() -> None:
    ground = {f: (5.0 + 1.0 * f, 0.5 * f) for f in range(15)}  # 33 m/s: no roll is that fast
    assert af.rolling_break(ground, FPS) is None


def test_find_contact_names_its_source_and_a_given_frame_wins() -> None:
    still = {f: (100.0 + (max(f - 10, 0)) * 9.0, 100.0) for f in range(20)}
    found = af.find_contact(still, {}, FPS)
    assert (found["source"], found["pre_contact"]) == ("rest", "still")
    flying = {f: (100.0 + 30.0 * f, 100.0) for f in range(20)}
    ground = {f: (5.0 + 1.1 * f, 0.0) for f in range(20)}
    found = af.find_contact(flying, ground, FPS)
    assert (found["frame"], found["source"], found["pre_contact"]) == (0, "first_frame", "unseen")
    given = af.find_contact(flying, ground, FPS, contact_frame=7)
    assert (given["frame"], given["source"], given["pre_contact"]) == (7, "manual", "rolling")


def test_a_rolling_ball_is_fitted_from_the_contact_with_its_incoming_line() -> None:
    positions = _rolling_positions()
    pixels = _pixels(positions)
    cameras = dict.fromkeys(range(N), L)
    ball_all = ball_3d(_ball_rows(pixels), cameras)
    result = af.auto_flight(ball_all, pixels, cameras, [], FPS, N)
    assert result is not None
    assert result["contact"]["source"] == "kink" and result["contact"]["pre_contact"] == "rolling"
    assert result["onset_frame"] in (12, 13)  # the kick lies between the last roll and the flight
    assert result["start"]["xyz"][:2] == pytest.approx(KICK_POINT, abs=0.3)  # not the roll's median
    assert result["barrier"] == {}  # no wall in open play
    assert result["incoming"]["speed_mps"] == pytest.approx(6.0, abs=0.6)
    assert result["incoming"]["turn_deg"] == pytest.approx(90.0, abs=10.0)  # +y in, +x out
    assert result["fit"]["rms_px"] < 3.0
    rows, record = af.auto_rows_and_record(result, pixels, N)
    assert record["pre_contact"] == "rolling" and record["contact_source"] == "kink"
    assert "p_wall" not in record["anchor_points"]
    before = [r for r in rows if r["frame"] < result["onset_frame"]]
    assert all(r["z_m"] == BALL_RADIUS for r in before)  # the roll stays on the ground
    assert before[0]["y_m"] == pytest.approx(positions[0, 1], abs=0.05)  # not frozen at the kick


def test_a_ball_flying_from_the_first_frame_starts_there(tmp_path) -> None:
    theta = list(SHOT)
    theta[5] = 0.0  # the kick is the first frame of the clip
    positions = _flight(theta)
    pixels = _pixels(positions)
    cameras = dict.fromkeys(range(N), L)
    result = af.auto_flight(ball_3d(_ball_rows(pixels), cameras), pixels, cameras, [], FPS, N)
    assert result is not None
    assert result["contact"]["source"] == "first_frame" and result["onset_frame"] == 0
    assert -1.0 / FPS - 1e-9 <= result["fit"]["kick_time_s"] <= 0.5 / FPS + 1e-9
    _, record = af.auto_rows_and_record(result, pixels, N)
    (tmp_path / "ball_flight.json").write_text(__import__("json").dumps(record), encoding="utf-8")
    read = af.flight_contact(tmp_path)
    assert read is not None and (read["frame"], read["pre_contact"]) == (0, "unseen")
    assert af.flight_contact(tmp_path / "missing") is None


# --- a panning broadcast camera (penalty clip, 08 Nov 2026) ---------------------------------------


def _panning_cameras(step_m: float = 0.08) -> dict[int, np.ndarray]:
    """The camera of every frame turns a little (its target slides along x): a resting ball moves
    in the image although it lies still on the pitch, as in a broadcast pan."""
    return {f: _camera([10.0, -38.0, 22.0], [50.0 - step_m * f, 0.0, 1.0]) for f in range(N)}


def test_ground_onset_and_stillness_ignore_the_image_motion() -> None:
    rest = {f: (41.45 + 0.03 * np.sin(f), 0.07) for f in range(20)}  # calibration noise only
    moved = {**rest, **{f: (41.45 + 0.9 * (f - 19), 0.07) for f in range(20, 30)}}
    assert af.is_still_ground(rest)
    assert af.kick_onset_ground(rest) is None
    assert af.kick_onset_ground(moved) == 20
    one_bad = dict(rest)
    one_bad[9] = (42.2, 0.07)  # one badly lifted frame is not a kick
    assert af.kick_onset_ground(one_bad) is None


def test_find_contact_reads_the_pitch_when_the_camera_pans() -> None:
    pixels = {f: (700.0 - 1.5 * f, 650.0) for f in range(20)}  # the pan moves the resting ball
    pixels.update({f: (700.0 + 60.0 * (f - 19), 640.0) for f in range(20, 30)})
    ground = dict.fromkeys(range(20), (41.48, 0.08))
    ground.update({f: (41.48 + 0.9 * (f - 19), 0.08) for f in range(20, 30)})
    assert af.find_contact(pixels, ground, FPS)["frame"] < 20  # the pixels alone see the pan
    found = af.find_contact(pixels, ground, FPS, panning=True)
    assert (found["frame"], found["source"], found["pre_contact"]) == (20, "rest", "still")


def test_camera_moves_tells_a_pan_from_a_fixed_camera() -> None:
    fixed = {f: dlt_projection_matrix(L) for f in range(N)}
    panning = {f: dlt_projection_matrix(Lf) for f, Lf in _panning_cameras().items()}
    point = (30.0, 0.0, BALL_RADIUS)
    assert not af.camera_moves(fixed, point)
    assert af.camera_moves(panning, point)


def test_a_panning_camera_does_not_move_the_kick_of_a_still_ball() -> None:
    theta = [30.0, 0.0, 22.0, 0.0, 7.0, 0.4, 0.0, 0.0, 0.0]  # kicked at 0.4 s (frame 12)
    positions = _flight(theta)
    cameras = _panning_cameras()
    projections = {f: dlt_projection_matrix(Lf) for f, Lf in cameras.items()}
    pixels = {}
    for f in range(N):
        h = projections[f] @ np.append(positions[f], 1.0)
        pixels[f] = (float(h[0] / h[2]), float(h[1] / h[2]))
    assert af.kick_onset(pixels) < 12  # the pan alone fakes an early kick in the image
    ball_all = ball_3d(_ball_rows(pixels), cameras)
    result = af.auto_flight(ball_all, pixels, cameras, [], FPS, N)
    assert result is not None
    assert result["contact"]["pre_contact"] == "still"
    assert result["onset_frame"] in (12, 13)
    assert result["fit"]["kick_time_s"] == pytest.approx(0.4, abs=1.5 / FPS)
    _, record = af.auto_rows_and_record(result, pixels, N)
    assert record["camera_moves"] is True


def _keeper_rows(frame: int, palm_xyz, person_id: int = 39) -> list[dict]:
    """A keeper's feet on the goal line and both palms (wrist, index and pinky tips) at ``palm_xyz``."""
    rows = []
    for name, xyz in (
        ("left-ankle", (52.0, -0.5, 0.1)),
        ("right-ankle", (52.0, 0.5, 0.1)),
        *[(f"{side}-{part}", palm_xyz) for side in ("left", "right")
          for part in ("wrist", "index-tip", "pinky-tip")],
    ):  # fmt: skip
        rows.append({"frame": frame, "person_id": person_id, "kpt_name": name,
                     "x_m": xyz[0], "y_m": xyz[1], "z_m": xyz[2]})  # fmt: skip
    return rows


def test_a_save_anchors_the_flight_on_the_keepers_hands_and_stops_it_there() -> None:
    # a penalty kicked at frame 12 from the spot; the keeper catches it at frame 24 in front of the line
    theta = [41.45, 0.0, 26.0, -3.0, 2.0, 0.4, 0.0, 0.0, 0.0]
    positions = _flight(theta)
    pixels = _pixels(positions)
    save = 24
    for f in range(save + 1, N):  # after the save the ball flies back: misleading pixels
        pixels[f] = tuple(_pix(positions[save] - (f - save) * np.array([0.3, 0.0, 0.0])))
    P = dlt_projection_matrix(L)
    centre = af.camera_centre(P)
    ray = positions[save] - centre
    ray /= np.linalg.norm(ray)
    toward_spot = np.array([41.45 - positions[save, 0], -positions[save, 1], 0.0])
    toward_spot /= np.linalg.norm(toward_spot)
    palms = (
        positions[save] - af.BALL_RADIUS_M * toward_spot + 0.3 * ray
    )  # the pose is 0.3 m too deep
    cameras = dict.fromkeys(range(N), L)
    ball_all = ball_3d(_ball_rows(pixels), cameras)
    result = af.auto_flight(ball_all, pixels, cameras, _keeper_rows(save, palms), FPS, N,
                            save_frame=save, keeper_id=39)  # fmt: skip
    assert result is not None
    arrival = result["arrival"]
    assert (arrival["type"], arrival["provenance"], arrival["frame"]) == (
        "keeper_hand",
        "auto",
        save,
    )
    end = np.asarray(result["fit"]["positions"][save])
    to_end = end - centre
    assert np.linalg.norm(to_end - (to_end @ ray) * ray) < 0.05  # on the ball's pixel ray
    assert (
        np.linalg.norm(end - positions[save]) < 0.45
    )  # the depth comes from the hands (0.3 m off)
    assert result["fit"]["kick_time_s"] == pytest.approx(0.4, abs=1.5 / FPS)
    assert arrival["diagnostics"]["ray_gap_m"] < 0.05  # the hands were already on the ray here
    rows, record = af.auto_rows_and_record(result, pixels, N)
    assert record["valid_until_frame"] == save and record["arrival_status"] == "keeper_hand"
    assert record["save_frame"] == save and record["shot_outcome"] == "save"
    assert all(np.isnan(r["x_m"]) for r in rows if r["frame"] > save)
    assert "p_arrival" in record["anchor_points"]
    assert "p0_along_m" in record["anchor_residuals"]
    # the uncertainty refit: the same anchor moved one sigma along the ray moves the end by ~sigma
    shifted = af.auto_flight(ball_all, pixels, cameras, _keeper_rows(save, palms), FPS, N,
                             save_frame=save, keeper_id=39, save_shift_m=0.4)  # fmt: skip
    end_shifted = np.asarray(shifted["fit"]["positions"][save])
    assert (end_shifted - end) @ ray == pytest.approx(0.4, abs=0.15)


def test_a_save_without_the_keepers_hands_still_stops_the_flight_at_the_save() -> None:
    theta = [41.45, 0.0, 26.0, -3.0, 2.0, 0.4, 0.0, 0.0, 0.0]
    pixels = _pixels(_flight(theta))
    cameras = dict.fromkeys(range(N), L)
    result = af.auto_flight(ball_3d(_ball_rows(pixels), cameras), pixels, cameras, [], FPS, N,
                            save_frame=24)  # fmt: skip
    assert result is not None and result["arrival"] is None
    rows, record = af.auto_rows_and_record(result, pixels, N)
    assert record["valid_until_frame"] == 24 and record["shot_outcome"] == "save"
    assert "keeper's hands" in record["quality"]
    assert all(np.isnan(r["x_m"]) for r in rows if r["frame"] > 24)
