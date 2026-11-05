"""Pass analysis heuristics from synthetic FreeKiki tracks."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from vaila import freekiki_pass as pa


def _write_pass_run(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    ball = ["frame,x_m,y_m,z_m,on_field\n"]
    ground = ["frame,person_id,x_m,y_m\n"]
    # frames 0-4: ball near passer (0,0); 5-14: ball travels to (10,0); 15-19: near receiver
    for f in range(20):
        if f <= 4:
            bx, by = 0.1, 0.0
        elif f <= 14:
            bx = 0.1 + (f - 4) * 1.0
            by = 0.0
        else:
            bx, by = 10.1, 0.0
        ball.append(f"{f},{bx},{by},0.11,True\n")
        ground.append(f"{f},12,0.0,0.0\n")  # passer stays
        ground.append(f"{f},8,10.0,0.0\n")  # receiver stays
    (folder / "ball_3d_on_field.csv").write_text("".join(ball), encoding="utf-8")
    (folder / "ground_tracks.csv").write_text("".join(ground), encoding="utf-8")
    (folder / "summary.json").write_text(json.dumps({"fps": 10.0}), encoding="utf-8")


def test_find_pass_window_release_before_receive() -> None:
    passer = np.zeros((20, 2))
    receiver = np.column_stack([np.full(20, 10.0), np.zeros(20)])
    ball = np.zeros((20, 2))
    for f in range(20):
        if f <= 4:
            ball[f] = [0.1, 0.0]
        elif f <= 14:
            ball[f] = [0.1 + (f - 4) * 1.0, 0.0]
        else:
            ball[f] = [10.1, 0.0]
    win = pa.find_pass_window(passer, receiver, ball, fps=10.0)
    assert win["release_frame"] is not None
    assert win["receive_frame"] is not None
    assert win["receive_frame"] > win["release_frame"]
    assert win["pass_chord_m"] is not None and win["pass_chord_m"] > 5.0


def test_run_pass_writes_json(tmp_path: Path) -> None:
    folder = tmp_path / "pass_3d"
    _write_pass_run(folder)
    result = pa.run_pass(folder, 12, 8, fps=10.0)
    assert result["kind"] == "pass"
    assert result["passer_id"] == 12 and result["receiver_id"] == 8
    assert result["ball"]["path_length_m"] > 8.0
    win = result["window"]
    assert win["receive_frame"] > win["release_frame"]
    path = folder / "data" / pa.ANALYSIS_NAME
    assert json.loads(path.read_text(encoding="utf-8"))["kind"] == "pass"


def test_passer_and_receiver_must_differ() -> None:
    import pytest

    with pytest.raises(ValueError, match="differ"):
        pa.analyze_pass("/tmp", 1, 1)


def _straight(n: int, start, v, fps: float) -> np.ndarray:
    t = np.arange(n) / fps
    return np.column_stack([start[0] + v[0] * t, start[1] + v[1] * t])


def test_pass_timing_receiver_already_running_gives_lower_bound() -> None:
    fps = 25.0
    n = 25
    passer = np.zeros((n, 2))
    ball = _straight(n, (0.5, 0.0), (15.0, 0.0), fps)  # already flying in frame 0
    receiver = _straight(n, (10.0, 3.0), (2.0, -5.0), fps)  # running at ~5.4 m/s from frame 0
    t = pa.pass_timing(passer, receiver, ball, fps, kick_time_s=-0.04)
    assert t["release_source"] == "flight_fit"
    assert math.isclose(t["release_t_s"], -0.04)
    run = t["receiver_run"]
    assert run["onset_before_release"] is True and run["onset_observed"] is False
    v0 = math.hypot(2.0, -5.0)
    expected = v0 / pa.A_MAX_MPS2 - run["t_first_s"]
    assert math.isclose(run["lead_lower_bound_s"], expected, rel_tol=1e-6)
    lo, hi = run["lead_lower_bound_range_s"]
    assert lo < run["lead_lower_bound_s"] < hi
    assert t["release_t_backextrap_s"] is not None and t["release_t_backextrap_s"] < 0


def test_pass_timing_observed_onset_after_release() -> None:
    fps = 20.0
    n = 30
    passer = np.zeros((n, 2))
    ball = np.zeros((n, 2))
    ball[5:] = _straight(n - 5, (0.2, 0.0), (10.0, 0.0), fps)
    receiver = np.tile([12.0, 0.0], (n, 1))
    for f in range(10, n):  # starts running at frame 10 (0.25 s after the release at 5)
        receiver[f] = [12.0 + 4.0 * (f - 9) / fps, 0.0]
    t = pa.pass_timing(passer, receiver, ball, fps, heuristic_release_frame=5)
    assert t["release_source"] == "heuristic"
    run = t["receiver_run"]
    assert run["onset_observed"] is True and run["onset_before_release"] is False
    assert math.isclose(run["onset_t_s"], (10 - 5) / fps)


def test_pass_timing_arrival_override_and_bounds() -> None:
    fps = 25.0
    n = 20
    passer = np.zeros((n, 2))
    ball = _straight(n, (0.5, 0.0), (12.0, 0.0), fps)
    receiver = np.tile([10.0, 0.5], (n, 1))
    auto = pa.pass_timing(passer, receiver, ball, fps)
    assert auto["arrival_source"] == "closest_approach"
    user = pa.pass_timing(passer, receiver, ball, fps, arrival_frame=19)
    assert user["arrival_source"] == "user" and user["arrival_frame"] == 19
    assert user["arrival_at_clip_end"] is True
    assert math.isclose(user["passer_to_arrival_m"], float(np.linalg.norm(ball[19])))
    import pytest

    with pytest.raises(ValueError, match="outside the clip"):
        pa.pass_timing(passer, receiver, ball, fps, arrival_frame=40)


def test_shot_link_from_setpiece_only(tmp_path: Path) -> None:
    shot = tmp_path / "kick_3d"
    (shot / "data").mkdir(parents=True)
    (shot / "data" / "setpiece.json").write_text(
        json.dumps({"ball_xy_m": [-42.0, -10.0], "distance_to_goal_m": 10.5, "goal_x_m": -52.45}),
        encoding="utf-8",
    )
    timing = {
        "arrival_xy": [-39.0, -10.0],
        "passer_at_release_xy": [-31.0, -24.0],
        "ball": {"v_arrival_mps": 10.0},
        "receiver_run": {"v_arrival_mps": 3.0},
    }
    link = pa.shot_link(shot, timing)
    assert "metrics_error" in link  # no flight fit in this folder
    assert link["shot_xy"] == [-42.0, -10.0]
    assert math.isclose(link["arrival_to_shot_m"], 3.0)
    assert math.isclose(link["distance_to_goal_m"], 10.5)
    assert link["gap_estimate_s"] == [0.3, 1.0]
    assert math.isclose(link["passer_to_shot_m"], math.hypot(11.0, 14.0))


def test_run_pass_writes_timeseries(tmp_path: Path) -> None:
    folder = tmp_path / "pass_ts"
    _write_pass_run(folder)
    result = pa.run_pass(folder, 12, 8, fps=10.0)
    ts = Path(result["timeseries"])
    assert ts.is_file()
    lines = ts.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("frame,t_rel_release_s,ball_x_m")
    assert len(lines) == 21
    assert result["fps_source"] == "user"
    assert "timing" in result and "_arrays" not in result
