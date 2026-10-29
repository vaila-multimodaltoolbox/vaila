"""The penalty adapter (vaila/freekiki_penalty.py): goal posts, goal frame, keeper pose and metrics.

The camera is synthetic: the goal mouth is a 100 px/m rectangle (the same layout as the pynalty tests),
so the ball and the keeper have known goal-plane coordinates that the metrics must recover.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vaila import freekiki_penalty as fp
from vaila.freekiki_pose_binocular import Keypoints
from vaila.pynalty_analysis import GoalGeometry, goal_plane_to_pixel
from vaila.pynalty_vision import LANDMARK_NAMES

GOAL_X = 52.45
CALIB = np.array(
    [[0.0, 244.0], [0.0, 0.0], [732.0, 0.0], [732.0, 244.0]]
)  # bottom-left ... as in pynalty


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


def test_goal_frame_is_the_first_crossing_after_the_kick_on_the_attacked_side() -> None:
    x = 10.0 + np.arange(60, dtype=float)  # 10 m to 69 m, crosses the goal line at frame 43
    ball = np.column_stack([x, np.zeros_like(x), np.full_like(x, 0.11)])
    assert fp.goal_frame(ball, GOAL_X, after=10) == 43
    assert fp.goal_frame(ball, -GOAL_X, after=10) is None  # the ball goes to the other side


def test_keypoints_map_to_the_pynalty_landmarks_and_missing_ones_stay_nan() -> None:
    kps = np.full((70, 2), np.nan)
    kps[15] = (123.0, 456.0)  # left-big-toe-tip
    arr = fp.landmark_array(kps, _names())
    foot = LANDMARK_NAMES.index("left_foot_index")
    np.testing.assert_allclose(arr[foot], (123.0, 456.0, 1.0))
    assert np.isnan(arr[LANDMARK_NAMES.index("left_eye_inner")]).all()  # not in the MHR70 model


def test_penalty_metrics_recovers_the_ball_and_the_keeper_on_the_goal_plane() -> None:
    goal = GoalGeometry()
    target_px = goal_plane_to_pixel(CALIB, np.array([[3.0, 1.0]]), goal)[
        0
    ]  # 3.0 m across, 1.0 m up
    rest = np.array([100.0, 100.0])
    n = 40
    ball_px = {}
    for f in range(n):
        progress = min(1.0, max(0.0, (f - 9) / 10.0))  # starts moving at frame 10, in the net at 19
        ball_px[f] = tuple(rest + (target_px - rest) * progress)
    ball_xyz = np.full((n, 3), np.nan)
    for f in range(n):
        ball_xyz[f] = (30.0 + (f - 10) * 2.5, 0.0, 0.11)  # across the goal line at frame 19

    kp = np.full((n, 70, 2), np.nan)
    for f in range(n):
        shift = 40.0 if f >= 14 else 0.0  # the keeper dives from frame 14
        kp[f, 5] = (500.0 + shift, 300.0)
        kp[f, 6] = (540.0 + shift, 300.0)
        kp[f, 9] = (500.0 + shift, 400.0)
        kp[f, 10] = (540.0 + shift, 400.0)
    keypoints = Keypoints(names=_names(), by_person={18: kp})
    keeper_boxes = {10: (480.0, 250.0, 520.0, 350.0), 19: (500.0, 250.0, 540.0, 350.0)}

    result = fp.penalty_metrics(
        field_rows=_post_rows(0, GOAL_X, 0.0, 732.0),
        ball_px=ball_px,
        ball_xyz=ball_xyz,
        keypoints=keypoints,
        keeper_boxes=keeper_boxes,
        keeper_sam=18,
        fps=59.94,
        goal_x=GOAL_X,
    )
    assert result["available"], result["reason"]
    assert result["kick_frame"] == 10
    assert result["goal_frame"] == 19
    assert result["gk_move_frame"] == 14
    assert abs(result["metrics"]["coord_x"] - 3.0) < 0.01
    assert abs(result["metrics"]["coord_z"] - 1.0) < 0.01


def test_a_ball_that_never_crosses_the_goal_line_is_reported_not_measured() -> None:
    ball_px = {f: (100.0 + (f * 10.0 if f >= 10 else 0.0), 100.0) for f in range(30)}
    ball_xyz = np.column_stack([np.full(30, 20.0), np.zeros(30), np.full(30, 0.11)])
    result = fp.penalty_metrics(
        field_rows=_post_rows(0, GOAL_X, 0.0, 732.0),
        ball_px=ball_px,
        ball_xyz=ball_xyz,
        keypoints=Keypoints(names=_names(), by_person={}),
        keeper_boxes={},
        keeper_sam=18,
        fps=59.94,
        goal_x=GOAL_X,
    )
    assert result["available"] is False
    assert "goal line" in result["reason"]


def test_keeper_is_read_from_the_player_map(tmp_path: Path) -> None:
    player_map = tmp_path / "player_map.csv"
    player_map.write_text(
        "player_id,role,camera,sam_id\n0,keeper,c1,18\n1,kicker,c1,16\n", encoding="utf-8"
    )
    assert fp.keeper_sam_id(player_map) == 18


def test_write_and_report_give_the_metrics_files(tmp_path: Path) -> None:
    result = {"available": True, "reason": "", "metrics": {"coord_x": 3.0, "coord_z": 1.0},
              "gk_dive": {}, "posts": {}, "calibration_px": CALIB.tolist(), "goal_x_m": GOAL_X,
              "kick_frame": 10, "goal_frame": 19, "gk_move_frame": 14, "gk_move_detected": True}  # fmt: skip
    paths = fp.write(tmp_path / "data", result)
    text = paths["csv"].read_text(encoding="utf-8")
    assert text.splitlines()[0] == "metric,value"
    assert "metrics.coord_x,3.0" in text
    assert paths["json"].exists()
    html = fp.report(tmp_path / "report", "c1.mp4", 59.94, result)
    assert Path(html["en"]).exists() and Path(html["pt"]).exists()
