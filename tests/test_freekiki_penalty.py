"""The penalty adapter (vaila/freekiki_penalty.py): goal posts, arrival, ball path, keeper pose and metrics.

The camera is a synthetic pinhole looking at the goal: the posts and the ball are projected by it, and
the ball truth comes from the same flight model the fit uses (``simulate_flight``). So the arrival, the
bend and the goal-mouth checks have known values the code must recover.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vaila import freekiki_penalty as fp
from vaila.freekiki_physics import simulate_flight
from vaila.freekiki_pose_binocular import Keypoints
from vaila.pynalty_analysis import GoalGeometry
from vaila.pynalty_vision import LANDMARK_NAMES

GOAL_X = 52.45
FPS = 59.94
N = 70
KICK = 4  # the ball starts to move at frame 4
CALIB = np.array(
    [[0.0, 244.0], [0.0, 0.0], [732.0, 0.0], [732.0, 244.0]]
)  # bottom-left ... as in pynalty


def _pinhole(center, target, f=1400.0, cx=960.0, cy=540.0) -> np.ndarray:
    """A 3x4 projection (pitch metres -> pixels) of a camera at ``center`` looking at ``target``."""
    c = np.asarray(center, dtype=float)
    forward = np.asarray(target, dtype=float) - c
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rot = np.vstack([right, down, forward])
    k = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
    return k @ np.hstack([rot, (-rot @ c)[:, None]])


CAMERA = _pinhole([25.0, -12.0, 7.0], [50.0, 0.0, 1.0])


def _project(P: np.ndarray, xyz) -> tuple[float, float]:
    h = P @ np.append(np.asarray(xyz, dtype=float), 1.0)
    return float(h[0] / h[2]), float(h[1] / h[2])


def _flight(spin_rps: float = 0.0, vy: float = 0.0, n: int = N, vx: float = 17.0) -> np.ndarray:
    """Ball centre per frame of a shot from the spot (the flight model of the fit), frame 0 at rest."""
    goal = GoalGeometry()
    theta = np.array(
        [
            GOAL_X - goal.penalty_distance,
            0.0,
            vx,
            vy,
            4.75,
            KICK / FPS,
            0.0,
            0.0,
            2 * np.pi * spin_rps,
        ]
    )
    return simulate_flight(theta, np.arange(n) / FPS, goal.ball_radius)


def _pixels_until_goal(pos: np.ndarray) -> tuple[dict[int, tuple[float, float]], int]:
    """The ball pixels up to the first frame across the goal line, and that frame."""
    last = int(np.where(pos[:, 0] >= GOAL_X)[0][0])
    return {f: _project(CAMERA, pos[f]) for f in range(last + 1)}, last


def _post_rows(
    frame: int, goal_x: float, u_left: float, u_right: float, side: str = "right"
) -> list[dict]:
    """The four posts of one goal at one frame, in the ``field_kps_3d.csv`` layout."""
    rows = []
    for name, y, z, u, v in (
        ("bottom_post_base", -3.66, 0.0, u_left, 244.0),
        ("bottom_post_top", -3.66, 2.44, u_left, 0.0),
        ("top_post_base", 3.66, 0.0, u_right, 244.0),
        ("top_post_top", 3.66, 2.44, u_right, 0.0),
    ):
        rows.append(
            {
                "frame": str(frame),
                "name": f"{side}_goal_{name}",
                "x_m": str(goal_x),
                "y_m": str(y),
                "z_m": str(z),
                "u_px": str(u),
                "v_px": str(v),
            }
        )
    return rows


def _camera_post_rows(P: np.ndarray, frame: int, goal_x: float) -> list[dict]:
    """The posts of one goal as the camera P sees them (the field model of a real run)."""
    rows = []
    for name, y, z in (
        ("bottom_post_base", -3.66, 0.0),
        ("bottom_post_top", -3.66, 2.44),
        ("top_post_base", 3.66, 0.0),
        ("top_post_top", 3.66, 2.44),
    ):
        u, v = _project(P, (goal_x, y, z))
        rows.append(
            {
                "frame": str(frame),
                "name": f"right_goal_{name}",
                "x_m": str(goal_x),
                "y_m": str(y),
                "z_m": str(z),
                "u_px": str(u),
                "v_px": str(v),
            }
        )
    return rows


def _names() -> list[str]:
    names = [""] * 70
    for k, name in (
        (0, "nose"),
        (5, "left-shoulder"),
        (6, "right-shoulder"),
        (9, "left-hip"),
        (10, "right-hip"),
        (15, "left-big-toe-tip"),
    ):
        names[k] = name
    return names


def test_goal_corners_are_the_attacked_posts_at_the_frame_nearest_the_kick() -> None:
    # frame 0 is shifted by 50 px to show that the frame nearest the kick (5) is the one used
    rows = _post_rows(0, GOAL_X, 50.0, 782.0) + _post_rows(5, GOAL_X, 0.0, 732.0)
    rows += _post_rows(5, -GOAL_X, 900.0, 1200.0, side="left")  # the other goal is ignored
    corners, posts = fp.goal_corners_px(rows, GOAL_X, frame=4)
    np.testing.assert_allclose(corners, CALIB)
    assert posts["frame"] == 5


def test_goal_corners_are_none_when_the_goal_is_not_in_the_field_model() -> None:
    rows = _post_rows(0, -GOAL_X, 0.0, 732.0, side="left")
    assert fp.goal_corners_px(rows, GOAL_X, frame=0) is None


def test_arrival_is_the_point_the_pixel_came_from_on_the_goal_line() -> None:
    point = np.array([GOAL_X, 1.2, 0.9])
    arrival = fp.arrival_on_goal_line(CAMERA, _project(CAMERA, point), GOAL_X)
    np.testing.assert_allclose(arrival, point, atol=1e-6)


def test_a_straight_shot_without_spin_is_straight_even_with_a_lob() -> None:
    pixels, last = _pixels_until_goal(_flight())
    cameras = dict.fromkeys(range(N), CAMERA)
    path = fp.penalty_path(pixels, cameras, kick=KICK, goal_x=GOAL_X, fps=FPS, n_frames=N)
    assert path["available"], path["reason"]
    assert path["goal_frame"] == last
    assert path["shape"] == "straight"
    assert path["max_dev_m"] < 0.05
    assert path["apex_z_m"] > 1.0  # the gravity arc is a height, not a curve
    truth = _flight()[last]
    np.testing.assert_allclose(path["arrival_m"][1:], truth[1:], atol=0.05)
    assert path["arrival_in_goal_mouth"] is True


def test_spin_bends_the_path_and_the_bow_side_follows_the_spin() -> None:
    right = _path_with_spin(6.0)
    left = _path_with_spin(-6.0)
    for path in (right, left):
        assert path["available"], path["reason"]
        assert path["shape"] == "curved"
        assert 0.2 < path["max_dev_m"] < 0.45
    assert right["bow_side"] == "right"
    assert left["bow_side"] == "left"


def _path_with_spin(spin_rps: float) -> dict:
    pixels, _last = _pixels_until_goal(_flight(spin_rps=spin_rps))
    cameras = dict.fromkeys(range(N), CAMERA)
    return fp.penalty_path(pixels, cameras, kick=KICK, goal_x=GOAL_X, fps=FPS, n_frames=N)


def test_an_arrival_wide_of_the_goal_mouth_is_flagged() -> None:
    pixels, _last = _pixels_until_goal(_flight(vy=6.0))
    cameras = dict.fromkeys(range(N), CAMERA)
    path = fp.penalty_path(pixels, cameras, kick=KICK, goal_x=GOAL_X, fps=FPS, n_frames=N)
    assert path["available"], path["reason"]
    assert abs(path["arrival_m"][1]) > GoalGeometry().width / 2
    assert path["arrival_in_goal_mouth"] is False


def test_a_path_needs_a_camera_after_the_kick() -> None:
    pixels, _last = _pixels_until_goal(_flight())
    path = fp.penalty_path(pixels, {}, kick=KICK, goal_x=GOAL_X, fps=FPS, n_frames=N)
    assert path["available"] is False
    assert "camera" in path["reason"]


def test_frames_past_the_clip_are_ignored() -> None:
    pixels, last = _pixels_until_goal(_flight())
    cameras = dict.fromkeys(range(N), CAMERA)
    extra_pixels = {**pixels, N + 5: (100.0, 100.0)}  # a row of ball.csv past the last frame
    extra_cameras = {**cameras, N + 5: CAMERA}
    path = fp.penalty_path(
        extra_pixels, extra_cameras, kick=KICK, goal_x=GOAL_X, fps=FPS, n_frames=N
    )
    assert path["available"], path["reason"]
    assert path["goal_frame"] == last


def test_keypoints_map_to_the_pynalty_landmarks_and_missing_ones_stay_nan() -> None:
    kps = np.full((70, 2), np.nan)
    kps[15] = (123.0, 456.0)  # left-big-toe-tip
    arr = fp.landmark_array(kps, _names())
    foot = LANDMARK_NAMES.index("left_foot_index")
    np.testing.assert_allclose(arr[foot], (123.0, 456.0, 1.0))
    assert np.isnan(arr[LANDMARK_NAMES.index("left_eye_inner")]).all()  # not in the MHR70 model


def test_penalty_metrics_recovers_the_arrival_and_the_keeper_on_the_goal_plane() -> None:
    pixels, last = _pixels_until_goal(_flight())
    kp = np.full((N, 70, 2), np.nan)
    for f in range(N):
        shift = 40.0 if f >= 14 else 0.0  # the keeper dives from frame 14
        kp[f, 5] = (500.0 + shift, 300.0)
        kp[f, 6] = (540.0 + shift, 300.0)
        kp[f, 9] = (500.0 + shift, 400.0)
        kp[f, 10] = (540.0 + shift, 400.0)
    keypoints = Keypoints(names=_names(), by_person={18: kp})
    keeper_boxes = dict.fromkeys(range(N), (480.0, 250.0, 520.0, 350.0))

    result = fp.penalty_metrics(
        field_rows=_camera_post_rows(CAMERA, 0, GOAL_X),
        ball_px=pixels,
        cameras=dict.fromkeys(range(N), CAMERA),
        keypoints=keypoints,
        keeper_boxes=keeper_boxes,
        keeper_sam=18,
        fps=FPS,
        n_frames=N,
        goal_x=GOAL_X,
    )
    assert result["available"], result["reason"]
    assert result["kick_frame"] in (KICK, KICK + 1)
    assert result["goal_frame"] == last
    assert result["gk_move_frame"] == 14
    assert result["path"]["shape"] == "straight"
    # pynalty's X runs from the left post in the image: the post at y=-3.66 is left when it has the
    # smaller u, so X = y + 3.66; otherwise X = 3.66 - y
    left_is_negative_y = (
        _project(CAMERA, (GOAL_X, -3.66, 0.0))[0] < _project(CAMERA, (GOAL_X, 3.66, 0.0))[0]
    )
    truth = _flight()[last]
    expected_x = truth[1] + 3.66 if left_is_negative_y else 3.66 - truth[1]
    assert abs(result["metrics"]["coord_x"] - expected_x) < 0.05
    assert abs(result["metrics"]["coord_z"] - truth[2]) < 0.05


def test_penalty_metrics_without_the_per_frame_camera_says_so() -> None:
    pixels, _last = _pixels_until_goal(_flight())
    result = fp.penalty_metrics(
        field_rows=_camera_post_rows(CAMERA, 0, GOAL_X),
        ball_px=pixels,
        cameras=None,
        keypoints=Keypoints(names=_names(), by_person={}),
        keeper_boxes={},
        keeper_sam=18,
        fps=FPS,
        n_frames=N,
        goal_x=GOAL_X,
    )
    assert result["available"] is False
    assert "DLT3D" in result["reason"]


def test_keeper_is_read_from_the_player_map(tmp_path: Path) -> None:
    player_map = tmp_path / "player_map.csv"
    player_map.write_text(
        "player_id,role,camera,sam_id\n0,keeper,c1,18\n1,kicker,c1,16\n", encoding="utf-8"
    )
    assert fp.keeper_sam_id(player_map) == 18


def test_write_and_report_give_the_metrics_files(tmp_path: Path) -> None:
    pixels, _last = _pixels_until_goal(_flight())
    path = fp.penalty_path(
        pixels, dict.fromkeys(range(N), CAMERA), kick=KICK, goal_x=GOAL_X, fps=FPS, n_frames=N
    )
    result = {"available": True, "reason": "", "metrics": {"coord_x": 3.0, "coord_z": 1.0},
              "gk_dive": {}, "posts": {}, "calibration_px": CALIB.tolist(), "goal_x_m": GOAL_X,
              "kick_frame": KICK, "goal_frame": path["goal_frame"], "gk_move_frame": 14,
              "gk_move_detected": True, "path": path}  # fmt: skip
    paths = fp.write(tmp_path / "data", result)
    text = paths["csv"].read_text(encoding="utf-8")
    assert text.splitlines()[0] == "metric,value"
    assert "metrics.coord_x,3.0" in text
    assert "path.rows" not in text  # the frame rows go to penalty_path.csv only
    assert "path.shape,straight" in text
    assert paths["json"].exists()
    trajectory = paths["path_csv"].read_text(encoding="utf-8").splitlines()
    assert trajectory[0].startswith("frame,x_m,y_m,z_m,horiz_dev_m,bow_left_m")
    assert len(trajectory) == path["last_frame"] - KICK + 2
    html = fp.report(tmp_path / "report", "c1.mp4", FPS, result)
    assert Path(html["en"]).exists() and Path(html["pt"]).exists()


def test_report_fills_flight_samples_and_video_block(tmp_path: Path) -> None:
    pixels, _last = _pixels_until_goal(_flight())
    path = fp.penalty_path(
        pixels, dict.fromkeys(range(N), CAMERA), kick=KICK, goal_x=GOAL_X, fps=FPS, n_frames=N
    )
    posts = {
        "left_post_y_m": -GoalGeometry().width / 2,
        "right_post_y_m": GoalGeometry().width / 2,
    }
    result = {
        "available": True,
        "reason": "",
        "metrics": {
            "coord_x": 3.0,
            "coord_z": 1.0,
            "ball_entry_x_m": 3.0,
            "ball_entry_z_m": 1.0,
            "flight_time_s": 0.5,
            "verdict_class": "reachable_standing",
            "verdict_label": "ok",
            "shot_outcome_label": "Goal",
        },
        "gk_dive": {},
        "posts": posts,
        "calibration_px": CALIB.tolist(),
        "goal_x_m": GOAL_X,
        "kick_frame": KICK,
        "goal_frame": path["goal_frame"],
        "gk_move_frame": 14,
        "path": path,
    }
    flight = fp.flight_path_array(result, FPS)
    assert flight is not None and flight.shape[1] == 4 and len(flight) >= 2
    media = tmp_path / "cam" / "report_tv" / "pt"
    media.mkdir(parents=True)
    (media / "animacao_lance.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42")
    html = fp.report(tmp_path / "cam" / "report_tv" / "penalty", "c1.mp4", FPS, result,
                     cam3d=tmp_path / "cam")  # fmt: skip
    body = Path(html["pt"]).read_text(encoding="utf-8")
    assert '"samples": [' in body and '"samples": []' not in body
    assert "<video" in body and "animacao_lance.mp4" in body


def test_a_shot_that_stays_short_of_the_goal_line_has_no_goal_frame() -> None:
    pos = _flight(vx=5.0)  # 6 m in the clip: the ball never reaches the line at 11 m from the spot
    pixels = {f: _project(CAMERA, pos[f]) for f in range(N)}
    cameras = dict.fromkeys(range(N), CAMERA)
    path = fp.penalty_path(pixels, cameras, kick=KICK, goal_x=GOAL_X, fps=FPS, n_frames=N)
    assert path["available"], path["reason"]
    assert path["goal_frame"] is None
    assert path["arrival_in_goal_mouth"] is False


def test_a_short_shot_is_unavailable_but_its_path_is_kept() -> None:
    pos = _flight(vx=5.0)
    pixels = {f: _project(CAMERA, pos[f]) for f in range(N)}
    result = fp.penalty_metrics(
        field_rows=_camera_post_rows(CAMERA, 0, GOAL_X),
        ball_px=pixels,
        cameras=dict.fromkeys(range(N), CAMERA),
        keypoints=Keypoints(names=_names(), by_person={}),
        keeper_boxes={},
        keeper_sam=18,
        fps=FPS,
        n_frames=N,
        goal_x=GOAL_X,
    )
    assert result["available"] is False
    assert "does not reach the goal line" in result["reason"]
    assert result["path"]["available"] is True
    assert result["path"]["goal_frame"] is None


def test_the_binocular_track_sets_the_verdict_when_there_are_two_cameras() -> None:
    pos = _flight(spin_rps=6.0)
    _pixels, last = _pixels_until_goal(pos)
    track = {f: pos[f] for f in range(last + 1)}  # the binocular ball, in metres, per frame
    path = fp.penalty_path({}, {}, kick=KICK, goal_x=GOAL_X, fps=FPS, n_frames=N, track=track)
    assert path["available"], path["reason"]
    assert path["source"] == "binocular"
    assert path["shape"] == "curved"
    assert path["bow_side"] == "right"
    assert path["goal_frame"] == last


def test_the_monocular_fit_is_labelled_as_provisional() -> None:
    assert _path_with_spin(0.0)["source"] == "monocular fit"


def _keeper_hips(move_to_y: float | None) -> dict[str, np.ndarray]:
    """The keeper's hips over 20 frames: still until frame 10, then moving to ``move_to_y`` by frame 14."""
    mid = np.zeros((20, 3))
    if move_to_y is not None:
        mid[10:15, 1] = np.linspace(0.2, move_to_y, 5)  # frame 10 is already 0.2 m away
        mid[15:, 1] = move_to_y
    return {"left-hip": mid + [0.0, 0.1, 0.0], "right-hip": mid - [0.0, 0.1, 0.0]}


def test_gk_partial_reports_the_onset_and_the_side_the_keeper_moves_to() -> None:
    result = fp.gk_partial(_keeper_hips(0.8), kick=5, last=14, fps=30.0, goal_x=GOAL_X)
    assert result["available"] and result["moved"] and result["partial"] is True
    assert result["onset_frame"] == 10
    # the goal is at +x: the keeper faces -x, so +y is his right
    assert abs(result["toward_right_m"] - 0.8) < 1e-6
    assert result["direction"] == "right"
    assert abs(result["mean_speed_mps"] - 0.8 / (9 / 30.0)) < 0.01


def test_gk_partial_reads_the_side_from_the_goal_sign() -> None:
    result = fp.gk_partial(_keeper_hips(0.8), kick=5, last=14, fps=30.0, goal_x=-GOAL_X)
    assert (
        result["direction"] == "left"
    )  # the goal is at -x: the keeper faces +x, so +y is his left


def test_gk_partial_says_when_the_keeper_stays_on_his_spot() -> None:
    result = fp.gk_partial(_keeper_hips(None), kick=5, last=14, fps=30.0, goal_x=GOAL_X)
    assert result["available"] and result["moved"] is False
    assert "stays within" in result["reason"]


def test_gk_partial_is_unavailable_without_the_hips() -> None:
    result = fp.gk_partial({}, kick=5, last=14, fps=30.0, goal_x=GOAL_X)
    assert result["available"] is False
    assert "hips" in result["reason"]


# --- a saved penalty (Lescano x Carlos Miguel, 08 Nov 2026): the hands stop the ball before the line ---


def _theta(vy: float = -1.5) -> np.ndarray:
    goal = GoalGeometry()
    return np.array(
        [GOAL_X - goal.penalty_distance, 0.0, 17.0, vy, 4.75, KICK / FPS, 0.0, 0.0, 0.0]
    )


def _dense_crossing(theta: np.ndarray) -> tuple[float, float, float]:
    times = np.arange(0.0, 2.0, 1e-4)
    pos = simulate_flight(theta, times, GoalGeometry().ball_radius)
    i = int(np.where(pos[:, 0] >= GOAL_X)[0][0])
    return float(pos[i, 1]), float(pos[i, 2]), float(times[i])


def _keeper_points(n: int, onset: int, save: int) -> dict[str, np.ndarray]:
    """A keeper on his line who shifts 0.8 m to -y from ``onset`` to ``save`` and drops 0.5 m."""
    pts = {name: np.full((n, 3), np.nan) for name in ("left-hip", "right-hip")}
    for f in range(n):
        w = float(np.clip((f - onset) / max(save - onset, 1), 0.0, 1.0))
        y, z = -0.8 * w, 1.0 - 0.5 * w
        pts["left-hip"][f] = (GOAL_X - 0.3, y - 0.1, z)
        pts["right-hip"][f] = (GOAL_X - 0.3, y + 0.1, z)
    return pts


def test_extrapolate_crossing_continues_the_flight_past_the_save() -> None:
    theta = _theta()
    y, z, t = _dense_crossing(theta)
    crossing = fp.extrapolate_crossing(theta, GOAL_X, t_from=0.5)
    assert crossing is not None
    assert abs(crossing["y_m"] - y) < 0.01 and abs(crossing["z_m"] - z) < 0.01
    assert abs(crossing["time_s"] - t) < 0.002
    wide = fp.extrapolate_crossing(_theta(vy=-30.0), GOAL_X, t_from=0.1)
    assert wide is not None and abs(wide["y_m"]) > 3.66  # it crosses the line, outside the posts
    slow = _theta()
    slow[2] = 3.0  # 3 m/s: the ball does not reach the line within the horizon
    assert fp.extrapolate_crossing(slow, GOAL_X, t_from=0.1) is None


def test_a_saved_penalty_reports_the_interception_and_the_would_be_crossing() -> None:
    theta = _theta()
    pos = simulate_flight(theta, np.arange(N) / FPS, GoalGeometry().ball_radius)
    save = int(np.where(pos[:, 0] >= GOAL_X - 0.6)[0][0])  # caught 0.6 m in front of the line
    flight = {"theta": theta.tolist(), "kick_time_s": KICK / FPS, "contact_frame": KICK + 1,
              "save_frame": save, "shot_outcome": "save"}  # fmt: skip
    track = {f: pos[f] for f in range(save + 1)}
    ball_px = {f: _project(CAMERA, pos[f]) for f in range(save + 1)}
    result = fp.penalty_metrics(
        field_rows=_camera_post_rows(CAMERA, save, GOAL_X),
        ball_px=ball_px,
        cameras=dict.fromkeys(range(N), CAMERA),
        keypoints=Keypoints(names=_names(), by_person={}),
        keeper_boxes={},
        keeper_sam=None,
        fps=FPS,
        n_frames=N,
        goal_x=GOAL_X,
        flight=flight,
        flight_track=track,
        save_frame=save,
        shot_outcome="save",
        keeper_points=_keeper_points(N, KICK - 2, save),
        gk_move_frame=KICK - 2,
    )
    assert result["available"] and result["event"] == "save"
    inter = result["interception"]
    assert np.allclose(inter["xyz"], pos[save], atol=1e-3)  # rounded to 0.1 mm
    assert inter["in_front_of_line_m"] == np.round(GOAL_X - pos[save, 0], 3)
    t_save = save / FPS - KICK / FPS
    assert abs(inter["time_s_after_kick"] - t_save) < 1e-4  # rounded to 0.1 ms
    spot = np.array([GOAL_X - 11.0, 0.0, 0.11])
    assert abs(inter["mean_speed_mps"] - np.linalg.norm(pos[save] - spot) / t_save) < 0.01
    y, z, _t = _dense_crossing(theta)
    would = result["counterfactual"]
    assert abs(would["y_m"] - y) < 0.01 and abs(would["z_m"] - z) < 0.01
    assert would["inside_goal_mouth"] is True and would["zone_label"]
    assert 0.0 <= would["hwr_accuracy_pct"] <= 100.0
    m = result["metrics"]
    assert m["shot_outcome"] == "save" and m["plane_source"] == "3d"
    assert abs(m["flight_time_s"] - t_save) < 1e-6
    assert abs(m["vel_ms"] - inter["mean_speed_mps"]) < 0.01
    hips = _keeper_points(N, KICK - 2, save)
    at = {f: (hips["left-hip"][f] + hips["right-hip"][f]) / 2 for f in (KICK + 1, save)}
    moved = np.hypot(*(at[save][1:] - at[KICK + 1][1:]))  # from the kick to the save, on the plane
    assert abs(m["gk_dist"] - moved) < 1e-6
    assert result["path"]["available"] and result["path"]["rows"][-1]["frame"] == save


# --- the speed of a saved penalty: one definition, its time base and its uncertainty ---------------


def _pan_pixels(t0: float, first: int, *, step: float = 66.0, drift: float = -1.0) -> dict:
    """Hand-marked pixels of a ball at rest that drifts with a panning camera, kicked at ``t0``."""
    px = {}
    for f in range(first - 6, first + 8):
        moved = max(f - t0, 0.0) * step
        px[f] = (700.0 + drift * (f - first) + moved, 650.0 - 0.3 * moved)
    return px


def test_the_kick_instant_is_read_from_the_marked_ball_pixels_alone() -> None:
    for t0 in (79.0, 79.24, 79.6):
        first = int(np.floor(t0)) + 1  # the first frame with the ball already moved
        found = fp.contact_from_pixels(_pan_pixels(t0, first), first)
        assert found is not None
        assert abs(found["frame"] - t0) < 0.05, (t0, found)
        assert found["low"] <= found["frame"] <= found["high"]
    # too few frames, or no rest before the kick: nothing to say
    assert fp.contact_from_pixels({80: (1.0, 1.0), 81: (2.0, 2.0)}, 80) is None
    assert fp.contact_from_pixels({}, 80) is None


def test_a_saved_penalty_has_one_speed_with_its_time_base_and_spread() -> None:
    theta = _theta()
    t0_frame = KICK + 0.3  # the contact falls between two frames
    theta[5] = t0_frame / FPS
    pos = simulate_flight(theta, np.arange(N) / FPS, GoalGeometry().ball_radius)
    save = int(np.where(pos[:, 0] >= GOAL_X - 0.6)[0][0])
    speed_mps = np.r_[0.0, np.linalg.norm(np.diff(pos, axis=0), axis=1) * FPS]
    flight = {"theta": theta.tolist(), "kick_time_s": float(theta[5]), "contact_frame": KICK + 1,
              "save_frame": save, "shot_outcome": "save", "peak_speed_mps": float(speed_mps.max()),
              "speed_mps": speed_mps.tolist()}  # fmt: skip
    track = {f: pos[f] for f in range(save + 1)}
    ball_px = {f: _project(CAMERA, pos[f]) for f in range(save + 1)}
    result = fp.penalty_metrics(
        field_rows=_camera_post_rows(CAMERA, save, GOAL_X),
        ball_px=ball_px,
        cameras=dict.fromkeys(range(N), CAMERA),
        keypoints=Keypoints(names=_names(), by_person={}),
        keeper_boxes={},
        keeper_sam=None,
        fps=FPS,
        n_frames=N,
        goal_x=GOAL_X,
        flight=flight,
        flight_track=track,
        save_frame=save,
        shot_outcome="save",
        keeper_points=_keeper_points(N, KICK - 2, save),
        gk_move_frame=KICK - 2,
    )
    sp = result["speed"]
    assert sp["fps"] == FPS and sp["frame_ms"] == round(1000.0 / FPS, 2)
    # time base: the sub-frame contact of the fit, next to the whole frames a viewer counts
    assert sp["kick"]["frame"] == round(t0_frame, 2) and sp["kick"]["counted_frame"] == KICK
    assert sp["kick"]["first_moved_frame"] == KICK + 1
    dt = save / FPS - t0_frame / FPS
    assert abs(sp["flight_time_s"] - dt) < 1e-4
    assert sp["flight_frames_counted"] == save - KICK
    assert abs(sp["flight_time_counted_s"] - (save - KICK) / FPS) < 1e-4
    # distance and the one mean speed: spot to the hands over the sub-frame flight time
    spot = np.array([GOAL_X - 11.0, 0.0, 0.11])
    dist = float(np.linalg.norm(pos[save] - spot))
    assert abs(sp["distance"]["spot_to_hands_m"] - dist) < 1e-3
    assert abs(sp["distance"]["path_m"] - dist) < 0.1 and sp["distance"]["path_m"] >= dist - 1e-3
    y, z, _t = _dense_crossing(theta)
    assert abs(sp["distance"]["to_line_m"] - np.hypot(np.hypot(11.0, y), z - 0.11)) < 0.02
    assert sp["distance"]["nominal_m"] == 11.0
    mean = sp["mean"]
    assert abs(mean["ms"] - dist / dt) < 0.01 and abs(mean["kmh"] - mean["ms"] * 3.6) < 0.06
    assert abs(mean["ms"] - result["interception"]["mean_speed_mps"]) < 0.01
    assert abs(mean["ms"] - result["metrics"]["vel_ms"]) < 0.01
    assert mean["kmh_counted"] < mean["kmh"]  # the counted frames count 0.3 frame more flight
    assert abs(mean["kmh_counted"] - dist / ((save - KICK) / FPS) * 3.6) < 0.06
    rel = np.hypot(0.3 / dist, np.hypot(0.25, 0.25) / FPS / dt)
    assert abs(mean["uncertainty_kmh"] - mean["kmh"] * rel) < 0.06
    assert mean["range_kmh"] == [
        round(mean["kmh"] - mean["uncertainty_kmh"], 1),
        round(mean["kmh"] + mean["uncertainty_kmh"], 1),
    ]
    # the model estimates are labelled as such and sit above the mean (drag slows the ball)
    exit_model = sp["exit_model"]
    assert abs(exit_model["launch_kmh"] - np.linalg.norm(theta[2:5]) * 3.6) < 0.06
    assert exit_model["launch_kmh"] > mean["kmh"] > sp["at_save_model"]["kmh"]
    assert abs(sp["at_save_model"]["kmh"] - speed_mps[save] * 3.6) < 0.06
    # the pixels alone place the kick where the fit put it
    check = sp["kick"]["pixel_check"]
    assert check is not None and abs(check["frame"] - t0_frame) < 0.2
    assert sp["kick"]["agrees_with_pixels"] is True
