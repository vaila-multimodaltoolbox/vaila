"""Planar kinematics for FreeKiki open-play dribble / pass analyses."""

from __future__ import annotations

import math

import numpy as np
import pytest

from vaila import freekiki_openplay_kinematics as kin


def test_path_length_of_axis_aligned_square() -> None:
    xy = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0], [0.0, 4.0], [0.0, 0.0]])
    assert abs(kin.path_length(xy) - 14.0) < 1e-9


def test_speed_and_accel_on_constant_velocity() -> None:
    fps = 10.0
    t = np.arange(0, 11) / fps
    xy = np.column_stack([2.0 * t, np.zeros_like(t)])  # 2 m/s along x
    v = kin.speed(xy, fps)
    assert abs(float(np.nanmean(v[1:])) - 2.0) < 1e-9
    a = kin.accel(v, fps)
    assert abs(float(np.nanmean(a[np.isfinite(a)]))) < 1e-9


def test_curvature_and_cod_on_right_angle() -> None:
    # go east then north: one ~90° COD
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [2.0, 2.0]])
    heads = kin.heading(xy)
    events = kin.cod_events(heads, min_deg=20.0)
    assert len(events) >= 1
    assert abs(events[0]["abs_deg"] - 90.0) < 1.0
    curv = kin.curvature_rad_per_m(xy)
    assert np.nanmax(curv) > 0.5  # rad/m at the corner


def test_interpersonal_distance_and_relative_speed() -> None:
    fps = 5.0
    a = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    b = np.array([[4.0, 0.0], [3.0, 0.0], [2.0, 0.0], [1.0, 0.0]])
    d = kin.interpersonal_distance(a, b)
    assert list(np.round(d, 6)) == [4.0, 2.0, 0.0, 2.0]
    approach = kin.pair_approach(a, b, fps)
    assert approach["min_distance_m"] == 0.0
    assert approach["min_frame"] == 2


def test_track_metrics_summary_keys() -> None:
    xy = np.column_stack([np.linspace(0, 10, 21), np.sin(np.linspace(0, math.pi, 21))])
    m = kin.track_metrics(xy, fps=20.0)
    assert m["path_length_m"] > 10.0
    assert m["v_max_mps"] is not None
    assert m["cod_count"] >= 0
    assert len(m["xy_m"]) == 21


def _line(n: int, v: float, fps: float) -> np.ndarray:
    t = np.arange(n) / fps
    return np.column_stack([v * t, np.zeros(n)])


def test_smooth_track_removes_outlier_and_keeps_edge_speed() -> None:
    fps = 25.0
    xy = _line(30, 5.0, fps)
    xy[12] += [0.0, 4.0]  # a 4 m detection jump
    out = kin.smooth_track(xy, fps)
    v = kin.speed(out, fps)[1:]
    assert np.all(np.abs(v - 5.0) < 0.5)
    # odd padding: the speed is not pulled to zero at the clip edges
    assert abs(v[0] - 5.0) < 0.3 and abs(v[-1] - 5.0) < 0.3


def test_smooth_track_fills_interior_gaps_only() -> None:
    xy = _line(20, 3.0, 20.0)
    xy[[0, 5, 6]] = np.nan
    out = kin.smooth_track(xy, 20.0)
    assert not np.isfinite(out[0]).any()
    assert np.isfinite(out[5:7]).all()


def _ball_files(folder, *, below: bool) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    z = -0.5 if below else 0.11
    (folder / "ball_3d_on_field.csv").write_text(
        f"frame,x_m,y_m,z_m,on_field\n0,1.0,0.0,0.11,True\n1,2.0,0.0,{z},True\n",
        encoding="utf-8",
    )
    (folder / "ball_3d.csv").write_text(
        "frame,cx,cy,score,x_m,y_m,z_m,on_field\n"
        "0,1,1,0.9,1.0,0.5,0.11,True\n1,1,1,0.2,9.0,9.0,0.11,True\n2,1,1,0.8,3.0,0.5,0.11,True\n",
        encoding="utf-8",
    )


def test_ball_source_auto_switches_to_ground_when_flight_is_below_pitch(tmp_path) -> None:
    good, bad = tmp_path / "good", tmp_path / "bad"
    _ball_files(good, below=False)
    _ball_files(bad, below=True)
    assert kin.ball_source(good)[1] == "flight"
    assert kin.ball_source(bad)[1] == "ground"
    assert kin.ball_source(good, "ground")[1] == "ground"
    ball = kin.read_ball_xy(bad)
    assert set(ball) == {0, 2}  # the low-score ground row is dropped
    assert ball[2] == (3.0, 0.5)


def test_detect_fps_source_reads_the_readme_video(tmp_path) -> None:
    import cv2

    video = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 24.0, (32, 32))
    for _ in range(3):
        writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
    writer.release()
    run = tmp_path / "run"
    run.mkdir()
    (run / "README.txt").write_text(f"FreeKiki 3D output\nvideo: {video}\n", encoding="utf-8")
    fps, source = kin.detect_fps_source(run)
    assert source == "video" and abs(fps - 24.0) < 1e-6
    missing = tmp_path / "missing"
    missing.mkdir()
    (missing / "README.txt").write_text("video: /nowhere/clip.mp4\n", encoding="utf-8")
    assert kin.detect_fps_source(missing) == (kin.DEFAULT_FPS, "default")


def _l_path(fps: float = 24.0, v: float = 4.0, n: int = 24) -> np.ndarray:
    """Run along +x, then turn 90 degrees left and run along +y at the same speed."""
    step = v / fps
    first = np.column_stack((np.arange(n) * step, np.zeros(n)))
    corner = first[-1]
    second = corner + np.column_stack((np.zeros(n), np.arange(1, n + 1) * step))
    return np.vstack((first, second))


def test_turn_between_measures_a_right_angle_and_speeds() -> None:
    xy = _l_path()
    turn = kin.turn_between(xy, 24.0, (0, 20), (27, 45))
    assert turn is not None
    assert turn["angle_deg"] == pytest.approx(90.0, abs=1e-6)
    assert turn["duration_s"] == pytest.approx(7 / 24)
    assert turn["v_in_mps"] == pytest.approx(4.0)
    assert turn["v_out_mps"] == pytest.approx(4.0)
    assert turn["angular_speed_deg_s"] == pytest.approx(90.0 / (7 / 24))
    assert turn["cod_zone"] == "high"
    assert turn["v_in_zone"] == 2  # 14.4 km/h


def test_turn_between_reversal_straight_and_gap() -> None:
    out_and_back = np.vstack(
        (
            np.column_stack((np.arange(10.0), np.zeros(10))),
            np.column_stack((np.arange(9.0, -1, -1), np.zeros(10))),
        )
    )
    reverse = kin.turn_between(out_and_back, 10.0, (0, 8), (11, 19))
    assert reverse is not None and reverse["abs_deg"] == pytest.approx(180.0)
    line = np.column_stack((np.arange(20.0), np.zeros(20)))
    straight = kin.turn_between(line, 10.0, (0, 8), (10, 19))
    assert straight is not None and straight["abs_deg"] == pytest.approx(0.0)
    line[8] = np.nan
    assert kin.turn_between(line, 10.0, (0, 8), (10, 19)) is None
    with pytest.raises(ValueError):
        kin.turn_between(line, 10.0, (0, 12), (10, 19))


def test_cod_and_speed_zones_follow_monteiro_2025() -> None:
    assert [kin.cod_zone(v) for v in (10, 30, 59.9, 60, 90, 400)] == [
        "very_low",
        "low",
        "low",
        "moderate",
        "high",
        "high",
    ]
    assert kin.cod_zone(None) is None
    assert [kin.speed_zone(v) for v in (5.9, 6, 14.9, 15, 20, 24.9, 25)] == [1, 2, 2, 3, 4, 4, 5]


def test_body_rotation_unwraps_a_half_turn() -> None:
    angles = np.radians(np.linspace(-100.0, -280.0, 17))  # 180 degrees, crossing -180
    left = np.column_stack((np.zeros(17), np.zeros(17)))
    right = np.column_stack((np.cos(angles), np.sin(angles)))
    right[3] = np.nan
    yaw = kin.body_yaw_deg(left, right)
    assert np.isnan(yaw[3])
    turn = kin.body_rotation(yaw, 24.0, 0, 16)
    assert turn is not None
    assert turn["angle_deg"] == pytest.approx(-180.0, abs=1e-6)
    assert turn["duration_s"] == pytest.approx(16 / 24)
    assert kin.body_rotation(yaw, 24.0, 3, 16) is None
