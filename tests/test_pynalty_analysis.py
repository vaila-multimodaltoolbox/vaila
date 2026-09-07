"""Unit tests for Pynalty pure-numpy analysis and report database helpers.

Update Date: 07 September 2026
Version: 0.3.129
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest

from vaila import pynalty
from vaila.pynalty_analysis import (
    DEFAULT_GK_HEIGHT_M,
    Anthropometrics,
    BallPath,
    BallPathPoint,
    GoalGeometry,
    ball_flight_distance,
    classify_shot_outcome,
    compute_penalty_metrics,
    dlt2d,
    fit_ball_flight_3d,
    flight_speed_at,
    gk_reach_envelope,
    goal_plane_to_pixel,
    goal_zone,
    interpolate_ball_path,
    is_convex_quadrilateral,
    nearest_body_point_to_ball,
    order_goal_corners,
    pixel_to_goal_plane,
    placement_index,
    reaction_classification,
    rec2d,
    save_verdict,
)
from vaila.pynalty_report import ReportContext, append_database, summary_row


def _square_calib_px(goal: GoalGeometry | None = None) -> np.ndarray:
    """Synthetic camera view of a FIFA goal as a clean rectangle in pixels."""
    goal = goal or GoalGeometry()
    # 100 px per metre, origin at bottom-left of the goal mouth in the image.
    scale = 100.0
    return np.array(
        [
            [0.0, goal.height * scale],
            [0.0, 0.0],
            [goal.width * scale, 0.0],
            [goal.width * scale, goal.height * scale],
        ],
        dtype=float,
    )


def test_dlt_round_trip_goal_corners() -> None:
    goal = GoalGeometry()
    calib = _square_calib_px(goal)
    A = dlt2d(goal.corner_coords(), calib)
    recovered = rec2d(A, calib)
    np.testing.assert_allclose(recovered, goal.corner_coords(), atol=1e-9)

    probe_real = np.array([[goal.mid_x, goal.mid_z], [1.0, 0.5], [6.5, 2.0]])
    probe_px = goal_plane_to_pixel(calib, probe_real, goal)
    back = pixel_to_goal_plane(calib, probe_px, goal)
    np.testing.assert_allclose(back, probe_real, atol=1e-8)


def test_goal_zone_and_placement_index() -> None:
    goal = GoalGeometry()
    row, col, idx, label = goal_zone(0.5, 0.3, goal)
    assert (row, col, idx) == (0, 0, 0)
    assert "Low" in label and "Left" in label

    row, col, idx, label = goal_zone(goal.mid_x, goal.mid_z, goal)
    assert (row, col) == (1, 1)
    assert idx == 4
    assert "Centre" in label

    corner = placement_index(0.2, 2.2, goal)
    centre = placement_index(goal.mid_x, goal.mid_z, goal)
    assert corner > centre
    assert 0.0 <= centre <= 100.0
    assert 0.0 <= corner <= 100.0


def test_fit_ball_flight_arrives_at_entry() -> None:
    goal = GoalGeometry()
    entry_x, entry_z, T = 6.0, 1.8, 0.45
    flight = fit_ball_flight_3d(entry_x, entry_z, T, n_samples=40, goal=goal)
    path = flight["path"]
    assert path.shape == (40, 4)
    np.testing.assert_allclose(path[0, 1:], [0.0, 0.0, goal.ball_radius], atol=1e-9)
    np.testing.assert_allclose(path[-1, 1], entry_x - goal.mid_x, atol=1e-9)
    np.testing.assert_allclose(path[-1, 2], goal.penalty_distance, atol=1e-9)
    np.testing.assert_allclose(path[-1, 3], entry_z, atol=1e-9)
    assert flight["speed0_ms"] > 0
    speeds = flight_speed_at(path, [0.0, T / 2.0, T])
    assert speeds.shape == (3,)
    assert np.all(np.isfinite(speeds))


def test_default_anthropometrics_enable_reach() -> None:
    raw = Anthropometrics()
    assert not raw.is_complete()
    resolved = raw.resolved(use_defaults=True)
    assert resolved.used_defaults is True
    assert resolved.gk_height_m == pytest.approx(DEFAULT_GK_HEIGHT_M)
    env = gk_reach_envelope(raw)
    assert env is not None
    assert env.radius_dive_m > env.radius_standing_m


def test_classify_shot_outcome_and_nearest_body() -> None:
    assert classify_shot_outcome("gol", ball_inside_goal=False)[0] == "goal"
    assert classify_shot_outcome("defesa", ball_inside_goal=True)[0] == "save"
    assert classify_shot_outcome(None, ball_inside_goal=False)[0] == "miss"
    assert classify_shot_outcome(None, ball_inside_goal=True)[0] == "goal"

    body = np.array([[0.0, 0.0], [2.0, 1.0], [5.0, 2.0]])
    pt, gap, idx = nearest_body_point_to_ball(body, (2.1, 1.1))
    assert idx == 1
    assert gap == pytest.approx(math.hypot(0.1, 0.1))
    np.testing.assert_allclose(pt, [2.0, 1.0])


def test_compute_penalty_metrics_uses_nearest_body_and_outcome() -> None:
    goal = GoalGeometry()
    calib = _square_calib_px(goal)
    # Three body points on the goal plane in pixels (100 px/m).
    body_px = [
        [goal.mid_x * 100.0, goal.mid_z * 100.0],
        [6.4 * 100.0, 0.6 * 100.0],  # nearer the entry
        [1.0 * 100.0, 1.0 * 100.0],
    ]
    metrics = compute_penalty_metrics(
        calib_pixels=calib,
        kick_frame=100,
        goal_frame=120,
        gk_move_frame=90,
        kick_ball_px=(goal.mid_x * 100.0, goal.height * 100.0),
        kick_gk_px=(goal.mid_x * 100.0, goal.mid_z * 100.0),
        goal_ball_px=(6.5 * 100.0, 0.5 * 100.0),
        goal_gk_px=(5.0 * 100.0, goal.mid_z * 100.0),
        fps=50.0,
        goal=goal,
        anthro=Anthropometrics(),  # defaults
        shot_outcome="save",
        gk_body_points_px=body_px,
        gk_body_landmark_names=["centre", "right_wrist", "left_hip"],
        defending_landmark_hint="right_wrist",
    )
    assert metrics["shot_outcome"] == "save"
    assert metrics["anthro_source"] == "default"
    assert metrics["gap_source"] == "defending_part"
    assert metrics["gap_landmark"] == "right_wrist"
    assert metrics["gap_m"] < metrics["gk_dist"] + 5  # sanity
    assert metrics["gk_height_m"] == pytest.approx(DEFAULT_GK_HEIGHT_M)

    anthro = Anthropometrics(gk_height_m=1.90)
    env = gk_reach_envelope(anthro)
    assert env is not None
    assert env.radius_dive_m > env.radius_standing_m

    reach_near = {
        "reachable_standing": True,
        "required_dive_speed_ms": 0.0,
        "elite_dive_speed_ms": 4.5,
        "time_margin_s": 0.2,
    }
    key, _ = save_verdict(reach_near, env)
    assert key == "reachable_standing"

    reach_far = {
        "reachable_standing": False,
        "required_dive_speed_ms": 9.0,
        "elite_dive_speed_ms": 4.5,
        "time_margin_s": -0.5,
    }
    key, _ = save_verdict(reach_far, env)
    assert key == "unsaveable"

    key, _ = save_verdict({}, None)
    assert key == "unknown"


def test_reaction_classification_bands() -> None:
    assert reaction_classification(-0.12)[0] == "anticipation"
    assert reaction_classification(0.0)[0] == "simultaneous"
    assert reaction_classification(0.18)[0] == "fast_reactive"
    assert reaction_classification(0.30)[0] == "reactive"
    assert reaction_classification(0.50)[0] == "late"


def test_interpolate_ball_path_fills_gaps() -> None:
    path = BallPath(
        points=[
            BallPathPoint(10, 100.0, 200.0, "manual"),
            BallPathPoint(14, 140.0, 240.0, "auto"),
        ]
    )
    filled = interpolate_ball_path(path, 10, 14)
    frames = [p.frame for p in filled.sorted_points()]
    assert frames == [10, 11, 12, 13, 14]
    mid = filled.by_frame()[12]
    assert mid.source == "interp"
    assert mid.x_px == pytest.approx(120.0)
    assert mid.y_px == pytest.approx(220.0)


def test_order_corners_and_convexity() -> None:
    goal = GoalGeometry()
    # order_goal_corners works in image space (Y down).
    calib = _square_calib_px(goal)
    shuffled_px = calib[[2, 0, 3, 1]]
    fixed_px = order_goal_corners(shuffled_px)
    np.testing.assert_allclose(fixed_px, calib, atol=1e-9)
    assert is_convex_quadrilateral(calib)
    crossed = np.array([[0, 0], [10, 10], [10, 0], [0, 10]], dtype=float)
    assert not is_convex_quadrilateral(crossed)


def test_compute_penalty_metrics_smoke() -> None:
    goal = GoalGeometry()
    calib = _square_calib_px(goal)
    # Ball enters upper-right third; keeper starts near centre.
    metrics = compute_penalty_metrics(
        calib_pixels=calib,
        kick_frame=100,
        goal_frame=120,
        gk_move_frame=90,
        kick_ball_px=(goal.mid_x * 100.0, goal.height * 100.0),
        kick_gk_px=(goal.mid_x * 100.0, goal.mid_z * 100.0),
        goal_ball_px=(6.5 * 100.0, 0.5 * 100.0),  # high Z in image = low in goal
        goal_gk_px=(5.0 * 100.0, goal.mid_z * 100.0),
        fps=50.0,
        goal=goal,
        anthro=Anthropometrics(gk_height_m=1.88, kicker_height_m=1.80),
    )
    assert metrics["flight_frames"] == 20
    assert metrics["flight_time_s"] == pytest.approx(0.4)
    assert metrics["ball_inside_goal"] is True
    assert metrics["dist"] == pytest.approx(
        ball_flight_distance(metrics["coord_x"], metrics["coord_z"], goal),
        rel=1e-6,
    )
    assert metrics["reaction_class"] == "anticipation"
    assert metrics["verdict_class"] in {
        "reachable_standing",
        "saveable",
        "saveable_late",
        "unsaveable",
    }
    assert "_flight_path" in metrics or "apex_height_m" in metrics


def test_toml_round_trip_via_app(tmp_path: Path) -> None:
    app = pynalty.PynaltyApp(show_wizard=False)
    app.fps = 59.94
    app.goal = GoalGeometry(width=7.32, height=2.44, penalty_distance=11.0)
    app.anthro = Anthropometrics(gk_height_m=1.91, gk_arm_span_m=1.95)
    app.step("gk_move").frame_idx = 10
    app.step("kick").frame_idx = 20
    app.step("kick").points = {"Ball": [100.0, 200.0], "GK": [300.0, 150.0]}
    app.step("goal").frame_idx = 40
    app.step("goal").points = {"Ball": [120.0, 180.0], "GK": [280.0, 160.0]}
    app.step("calibration").points = {"points": _square_calib_px().tolist()}
    app.set_ball_path(
        BallPath(
            points=[
                BallPathPoint(20, 100.0, 200.0, "manual"),
                BallPathPoint(30, 110.0, 190.0, "auto"),
            ]
        )
    )

    data = app.to_data()
    # Homogeneous parallel arrays must be TOML-safe (no mixed-type rows).
    bp = next(e for e in data["events"] if e["key"] == "ball_path")
    assert bp["points"]["path_frames"] == [20, 30]
    assert all(isinstance(x, float) for x in bp["points"]["path_x"])

    restored = pynalty.PynaltyApp(show_wizard=False)
    assert restored.load_from_data(data) is True
    assert restored.fps == pytest.approx(59.94)
    assert restored.anthro.gk_height_m == pytest.approx(1.91)
    assert restored.step("kick").frame_idx == 20
    assert restored.step("kick").points["Ball"] == [100.0, 200.0]
    path = restored.ball_path()
    assert [p.frame for p in path.sorted_points()] == [20, 30]
    assert path.by_frame()[30].source == "auto"


def test_database_append_idempotent(tmp_path: Path) -> None:
    goal = GoalGeometry()
    metrics = {
        "kick_frame": 250,
        "goal_frame": 278,
        "dist": 11.2,
        "vel_ms": 24.0,
        "vel_kmh": 86.4,
        "coord_x": 2.1,
        "coord_z": 1.4,
        "zone_label": "Middle Left",
        "reaction_class": "anticipation",
        "verdict_class": "saveable_late",
    }
    ctx = ReportContext(
        video_path="/tmp/clip_a.mp4",
        fps=60.0,
        metrics=metrics,
        goal=goal,
        generated_at="2026-09-07 12:00:00",
    )
    db = tmp_path / "pynalty_database.csv"
    append_database(str(db), ctx)
    append_database(str(db), ctx)  # same video + kick_frame → replace, not duplicate

    with open(db, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["video_name"] == "clip_a.mp4"
    assert rows[0]["kick_frame"] == "250"

    ctx2 = ReportContext(
        video_path="/tmp/clip_b.mp4",
        fps=60.0,
        metrics={**metrics, "kick_frame": 100, "dist": 10.5},
        goal=goal,
        generated_at="2026-09-07 12:01:00",
    )
    append_database(str(db), ctx2)
    with open(db, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert summary_row(ctx)["video_name"] == "clip_a.mp4"
