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
