"""Tests of vaila.kick_measure: per-frame cameras, heights, line plane, spin, arcs, series.

Update Date: 23 November 2026
Version: 0.4.19

Everything is synthetic (a known broadcast camera with a fixed optical centre, panning and zooming
frame by frame; a kick folder written into ``tmp_path``): no external data, no GPU, no window.
"""

from __future__ import annotations

import dataclasses
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from vaila import kick_measure as km
from vaila.freekiki_physics import simulate_flight

W, H = 1920, 1080
CENTRE = np.array([0.0, -75.0, 22.0])
FPS = 30.0
N_FRAMES = 30
STEM = "maincam"
TS = "20261123_100000"


def _template() -> np.ndarray:
    return km.load_field_template()


def _true_camera(frame: int) -> np.ndarray:
    """Camera of the synthetic broadcast: fixed centre, pan and zoom change every frame."""
    target = np.array([-30.0 + 2.0 * frame, 0.0, 0.0])
    zc = (target - CENTRE) / np.linalg.norm(target - CENTRE)
    xc = np.cross(zc, [0.0, 0.0, 1.0])
    xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    R = np.array([xc, yc, zc])
    f = 2200.0 + 40.0 * frame
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    return K @ np.c_[R, -R @ CENTRE]


def _visible_keypoints(P: np.ndarray, noise: float, rng) -> dict[int, tuple[float, float, float]]:
    tmpl = _template()
    uv = km.project_points(P, tmpl)
    out = {}
    for i, (u, v) in enumerate(uv):
        if 0 < u < W and 0 < v < H:
            du, dv = rng.normal(0.0, noise, 2) if noise else (0.0, 0.0)
            out[i] = (float(u + du), float(v + dv), 0.9)
    return out


def _normalised(P: np.ndarray) -> np.ndarray:
    return P / P[2, 3]  # the FreeKiki DLT3D has P[2, 3] = 1


def _calibration(dlt_every: int = 3, noise: float = 0.3):
    rng = np.random.default_rng(7)
    truth = {f: _true_camera(f) for f in range(N_FRAMES)}
    dlt = {f: _normalised(truth[f]) for f in range(0, N_FRAMES, dlt_every)}
    kps = {f: _visible_keypoints(truth[f], noise, rng) for f in range(N_FRAMES)}
    return truth, dlt, kps


# ----------------------------------------------------------------------------------------------
# 1. Camera: fixed centre, rotation + zoom per frame
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("noise", "tol_px"), [(0.0, 0.01), (0.3, 1.5)])
def test_fixed_centre_solver_recovers_rotation_zoom_centre(noise, tol_px):
    truth, dlt, kps = _calibration(noise=noise)
    # frame 7: too few points (approximate); frame 8: none at all
    kps[7] = dict(list(kps[7].items())[:2])
    kps[8] = {}
    track = km.CameraTrack.from_calibration(
        dlt, kps, _template(), N_FRAMES, principal_point=(W / 2, H / 2)
    )
    assert track.centre is not None
    assert np.linalg.norm(track.centre - CENTRE) < 1e-6

    grid = np.array([[x, y, 0.0] for x in np.linspace(-30, 30, 7) for y in np.linspace(-20, 20, 5)])
    solved = 0
    for f in range(N_FRAMES):
        info = track.info(f)
        if f in dlt:
            assert info.source == km.SOURCE_DLT3D
            assert np.array_equal(info.matrix, dlt[f])  # calibrated frames are kept untouched
            continue
        if f in (7, 8):
            assert info.approximate and info.source == km.SOURCE_APPROX
            assert info.copied_from is not None and info.copied_from in dlt
            continue
        assert info.source == km.SOURCE_SOLVED
        solved += 1
        # projection error over the pitch, in pixels (only points that are in front and in view)
        uv_true = km.project_points(truth[f], grid)
        uv_est = track.project(f, grid)
        seen = (np.abs(uv_true[:, 0] - W / 2) < W) & (np.abs(uv_true[:, 1] - H / 2) < H)
        assert np.hypot(*(uv_est - uv_true)[seen].T).max() < tol_px
        # zoom and rotation
        f_true, R_true = km._decompose(truth[f])
        assert info.focal_px == pytest.approx(f_true, rel=2e-3 if noise else 1e-6)
        assert info.rvec is not None
        R_est = Rotation.from_rotvec(info.rvec).as_matrix()
        assert math.degrees(np.arccos(np.clip((np.trace(R_est @ R_true.T) - 1) / 2, -1, 1))) < 0.05
        assert info.rms_px < 1.0
        # ground point of some pixels against the true camera: < 5 cm
        for u, v in ((600, 700), (960, 540), (1400, 800), (300, 900)):
            a = track.ground_point(f, u, v)
            b = km.ground_point(truth[f], u, v)
            assert np.hypot(*(a[:2] - b[:2])) < 0.05
    assert solved == N_FRAMES - len(dlt) - 2
    counts = track.counts()
    assert counts == {"dlt3d": len(dlt), "solved": solved, "approximate": 2}
    assert track.approximate_frames() == [7, 8]
    assert track.is_approximate(7) and not track.is_approximate(6)


def test_camera_track_given_matrices_and_outside_range():
    P = _true_camera(0)
    track = km.CameraTrack.from_matrices({0: P, 10: _true_camera(10)}, n_frames=12)
    assert track.source(0) == km.SOURCE_GIVEN
    assert np.array_equal(track.matrix(-5), P)  # nearest frame outside the range
    assert track.is_approximate(5) is True  # a hole copies the nearest and says so
    assert track.info(11).copied_from == 10


def test_ground_point_sign_agnostic_and_horizon():
    P = _true_camera(5)
    X = np.array([12.0, 7.0, 0.0])
    u, v = km.project_points(P, X)[0]
    for sign in (1.0, -1.0):  # a DLT matrix may carry either overall sign
        g = km.ground_point(sign * P, u, v, 0.0)
        assert np.allclose(g, X, atol=1e-6)
    z = km.ground_point(P, u, v, 1.5)
    assert z[2] == 1.5 and np.isfinite(z).all()
    assert np.isnan(km.ground_point(P, 960, -5000, 0.0)[:2]).all()  # above the horizon


# ----------------------------------------------------------------------------------------------
# 2. Operations
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_height_base_top_pole_of_one_metre(sign):
    P = sign * _true_camera(10)
    for base in ([5.0, 3.0], [-20.0, 10.0], [30.0, -8.0]):
        b, t = np.array([*base, 0.0]), np.array([*base, 1.0])
        buv = km.project_points(P, b)[0]
        tuv = km.project_points(P, t)[0]
        h = km.height_base_top(P, buv, tuv)
        assert h == pytest.approx(1.0, abs=0.01)
    # an elevated base (a pole on a 2 m stand)
    b, t = np.array([4.0, 4.0, 2.0]), np.array([4.0, 4.0, 3.0])
    buv, tuv = km.project_points(P, b)[0], km.project_points(P, t)[0]
    assert km.height_base_top(P, buv, tuv, base_z=2.0) == pytest.approx(1.0, abs=0.01)


def test_height_base_top_through_the_track():
    track = km.CameraTrack.from_matrices({3: _true_camera(3)}, n_frames=4)
    b, t = np.array([8.0, -2.0, 0.0]), np.array([8.0, -2.0, 1.0])
    buv, tuv = track.project(3, b)[0], track.project(3, t)[0]
    assert track.height_base_top(3, buv, tuv) == pytest.approx(1.0, abs=0.01)


def test_line_plane_against_simulated_flight():
    # Without spin the drag is anti-parallel to the velocity: the flight stays in the vertical
    # plane of the kick direction, so LinePlane must give (s, z) exactly.
    x0, y0, vx, vy, vz = -20.0, -5.0, 22.0, 6.0, 11.0
    theta = np.array([x0, y0, vx, vy, vz, 0.0, 0.0, 0.0, 0.0])
    frames = np.arange(0, 60)
    xyz = simulate_flight(theta, frames / FPS)
    assert np.isfinite(xyz).all()
    P = _true_camera(15)
    track = km.CameraTrack.from_matrices({0: P}, n_frames=60)
    heading = np.array([vx, vy]) / np.hypot(vx, vy)
    plane = km.LinePlane(track, 0, [x0, y0], [x0 + 60 * heading[0], y0 + 60 * heading[1]])
    assert plane.length == pytest.approx(60.0)
    s_true = (xyz[:, :2] - [x0, y0]) @ heading
    for i in range(0, 60, 3):
        uv = track.project(0, xyz[i])[0]
        s, z = plane.locate(0, uv)
        assert s == pytest.approx(s_true[i], abs=1e-6)
        assert z == pytest.approx(xyz[i, 2], abs=1e-6)
    # the same from two clicks on the grass: the plane is the one of the pixels
    a, b = np.array([x0, y0, 0.0]), np.array([x0 + 60 * heading[0], y0 + 60 * heading[1], 0.0])
    p2 = km.LinePlane.from_pixels(track, 0, track.project(0, a)[0], track.project(0, b)[0])
    assert p2.locate(0, track.project(0, xyz[30])[0])[1] == pytest.approx(xyz[30, 2], abs=1e-6)
    # a 2 m lateral shift of the plane changes the height by a few decimetres, not metres
    uv = track.project(0, xyz[30])[0]
    assert 0.0 < abs(plane.lateral_sensitivity(0, uv, 2.0)) < 1.5
    # the guide line of a given height is drawn at that height
    guide = plane.height_line(0, 2.44, 0.0, 30.0)
    assert guide.shape == (2, 2)
    back = plane.locate(0, guide[1])
    assert back[0] == pytest.approx(30.0, abs=1e-6) and back[1] == pytest.approx(2.44, abs=1e-6)


def test_line_plane_exact_parabola_series_and_kinematics():
    # an exact parabola in (s, z): constant speed along s, g downward in z
    t = np.arange(0, 40) / FPS
    s_true, z_true = 20.0 * t, 11.0 * t - 0.5 * 9.81 * t**2 + 0.11
    pos = np.c_[s_true, z_true]
    v = km.velocity_series(pos, FPS)
    a = km.acceleration_series(pos, FPS)
    assert np.allclose(v[:, 0], 20.0, atol=1e-9)
    assert np.allclose(v[:, 1], 11.0 - 9.81 * t, atol=1e-9)
    assert np.allclose(a[:, 1], -9.81, atol=1e-8)
    assert np.allclose(a[:, 0], 0.0, atol=1e-8)
    # Savitzky-Golay of order 2 reproduces an exact parabola
    vs = km.velocity_series(pos, FPS, smooth_window=7)
    as_ = km.acceleration_series(pos, FPS, smooth_window=7)
    assert np.allclose(vs[:, 1], 11.0 - 9.81 * t, atol=1e-8)
    assert np.allclose(as_[:, 1], -9.81, atol=1e-6)
    speed = km.speed_series(pos, FPS)
    assert speed[5] == pytest.approx(math.hypot(20.0, 11.0 - 9.81 * t[5]))
    fit = km.fit_parabola(pos)
    assert fit.a == pytest.approx(-0.5 * 9.81 / 20.0**2, rel=1e-9)
    assert fit.apex_s == pytest.approx(20.0 * 11.0 / 9.81, rel=1e-9)
    assert fit.rms < 1e-9
    assert fit.z_at(fit.apex_s) == pytest.approx(fit.apex_z)


def test_series_keep_nan_holes_and_handle_short_input():
    pos = np.arange(10, dtype=float) * 2.0  # 2 m per frame
    pos[4] = np.nan
    v = km.velocity_series(pos, 10.0)  # 20 m/s
    assert np.isnan(v[4]) and np.allclose(np.delete(v, 4), 20.0)
    assert km.velocity_series(np.array([1.0]), 10.0).shape == (1,)
    assert np.isnan(km.acceleration_series(np.array([1.0, 2.0]), 10.0)).all()
    with pytest.raises(ValueError):
        km.velocity_series(pos, 0.0)


def test_spin_rate_five_rps_at_thirty_fps():
    marks = [3, 9, 15, 21, 27, 33]  # the same mark every 6 frames = 5 turns in 1 s
    r = km.spin_rate(marks, 30.0)
    assert r.turns == 5
    assert r.duration_s == pytest.approx(1.0)
    assert r.rps == pytest.approx(5.0)
    assert r.rpm == pytest.approx(300.0)
    assert r.uncertainty_s == pytest.approx(1 / 30.0)
    # marks read to the nearest frame of a 4.7 rps ball: within the stated +-1/fps
    true_rps, fps = 4.7, 30.0
    marks = [round(k * fps / true_rps + 2.4) for k in range(9)]
    r = km.spin_rate(marks, fps)
    assert abs(r.duration_s - 8 / true_rps) <= 1 / fps
    assert abs(r.rps - true_rps) <= 1.25 * r.rps_uncertainty
    assert r.rpm == pytest.approx(60 * r.rps)
    with pytest.raises(ValueError):
        km.spin_rate([4], 30.0)
    with pytest.raises(ValueError):
        km.spin_rate([4, 4], 30.0)


def test_event_time_and_mean_speed():
    assert km.event_time(10, 40, 30.0) == pytest.approx(1.0)
    assert km.event_time(40, 10, 30.0) == pytest.approx(-1.0)
    assert km.event_time_uncertainty(30.0) == pytest.approx(1 / 30.0)
    assert km.mean_speed([0, 0, 0], [3, 4, 0], 10, 40, 30.0) == pytest.approx(5.0)
    assert km.mean_speed([0, 0], [30, 40], 40, 10, 30.0) == pytest.approx(50.0)
    with pytest.raises(ValueError):
        km.mean_speed([0, 0], [1, 1], 5, 5, 30.0)


def test_arc_of_a_known_circle():
    radius, centre = 20.0, np.array([3.0, -4.0])
    ang = np.radians(np.linspace(10.0, 70.0, 41))
    pts = centre + radius * np.c_[np.cos(ang), np.sin(ang)]
    fit = km.fit_arc(pts)
    assert fit.radius == pytest.approx(radius, abs=1e-6)
    assert fit.centre == pytest.approx(tuple(centre), abs=1e-6)
    assert fit.angle_deg == pytest.approx(60.0, abs=1e-6)
    assert fit.arc_length == pytest.approx(radius * math.pi / 3, abs=1e-5)
    assert fit.sagitta == pytest.approx(radius * (1 - math.cos(math.radians(30))), abs=1e-6)
    assert fit.chord == pytest.approx(2 * radius * math.sin(math.radians(30)), abs=1e-6)
    assert fit.rms < 1e-6
    assert km.path_length(pts) == pytest.approx(fit.arc_length, rel=1e-3)
    # clockwise order gives the same covered angle; noise of 1 mm barely moves the radius
    noisy = pts[::-1] + np.random.default_rng(1).normal(0, 1e-3, pts.shape)
    fit2 = km.fit_arc(noisy)
    assert fit2.angle_deg == pytest.approx(60.0, abs=0.5)
    assert fit2.radius == pytest.approx(radius, abs=0.1)
    # a straight line has no radius; too few points is an error
    line = km.fit_arc([[0, 0], [1, 1], [2, 2], [3, 3]])
    assert math.isinf(line.radius) and line.angle_rad == 0.0 and line.sagitta == 0.0
    with pytest.raises(ValueError):
        km.fit_arc([[0, 0], [1, 1]])


def test_path_length():
    assert km.path_length([[0, 0], [3, 4], [3, 10]]) == pytest.approx(11.0)
    assert km.path_length([[0, 0, 0], [1, 2, 2]]) == pytest.approx(3.0)
    assert km.path_length([[0, 0], [np.nan, 1], [3, 4]]) == pytest.approx(5.0)
    assert km.path_length([[1, 1]]) == 0.0


# ----------------------------------------------------------------------------------------------
# 3. A kick_analysis folder written by hand
# ----------------------------------------------------------------------------------------------


@pytest.fixture
def kick_folder(tmp_path: Path) -> dict:
    truth, dlt, kps = _calibration()
    clip = tmp_path / "clip"
    clip.mkdir()
    video = clip / f"{STEM}.mp4"  # never created: fps comes from the README
    run = clip / f"kick_analysis_{TS}"
    predict = run / "work" / STEM / f"freekiki_predict_{STEM}_{TS}"
    predict.mkdir(parents=True)
    rows = []
    for f in range(N_FRAMES):
        if f in dlt:
            P = dlt[f]
            rows.append([f, *P[0], *P[1], *P[2, :3]])
        else:
            rows.append([f, *([np.nan] * 11)])
    cols = ["frame"] + [f"L{i}" for i in range(1, 12)]
    pd.DataFrame(rows, columns=pd.Index(cols)).to_csv(predict / f"{STEM}.dlt3d", index=False)
    raw = []
    for f in range(N_FRAMES):
        row: dict = {"frame": f, "box_conf": 0.9}
        for i in range(49):
            u, v, c = kps[f].get(i, (0.0, 0.0, 0.0))
            row[f"p{i}_x"], row[f"p{i}_y"], row[f"p{i}_conf"] = u, v, c
        raw.append(row)
    pd.DataFrame(raw).to_csv(predict / "field_kps_raw.csv", index=False)

    threed = run / f"{STEM}_3d_{TS}"
    threed.mkdir()
    ball_xyz = np.array([[-10.0 + 1.0 * f, -3.0, 0.11 + 0.05 * f] for f in range(N_FRAMES)])
    uv = np.array([km.project_points(truth[f], ball_xyz[f])[0] for f in range(N_FRAMES)])
    pd.DataFrame({"frame": range(N_FRAMES), "cx": uv[:, 0], "cy": uv[:, 1]}).to_csv(
        threed / "ball.csv", index=False
    )
    pd.DataFrame(
        {"frame": [0, 1], "person_id": [3, 3], "x_m": [1.0, 1.5], "y_m": [2.0, 2.5]}
    ).to_csv(threed / "ground_tracks.csv", index=False)
    sam = run / "work" / STEM / "sam3dinov3" / "p1" / STEM / "sam3"
    sam.mkdir(parents=True)
    foot = np.array([5.0, -4.0, 0.0])
    fuv = km.project_points(truth[0], foot)[0]
    pd.DataFrame(
        {
            "frame": [0, 0, 1],
            "obj_id": [3, 4, 3],
            "x_px": [fuv[0] - 20, 500.0, 600.0],
            "y_px": [fuv[1] - 90, 400.0, 410.0],
            "w_px": [40.0, 40.0, 40.0],
            "h_px": [90.0, 100.0, 100.0],
        }
    ).to_csv(sam / "sam_tracks.csv", index=False)
    status = {
        "ts": TS,
        "stages": {
            f"detect:{STEM}": {
                "status": "ok",
                "inputs": {"video": {"path": str(video)}},
                "result": {"predict": str(predict)},
            }
        },
        "options": {
            "videos": [str(video)],
            "set_piece": "open_play",
            "analysis": None,
            "sync": None,
        },  # fmt: skip
    }
    (run / "pipeline_status.json").write_text(json.dumps(status), encoding="utf-8")
    (run / "README.txt").write_text(f"created: x\nfps: {FPS} | kicker id: 1\n", encoding="utf-8")
    return {"run": run, "predict": predict, "threed": threed, "truth": truth, "dlt": dlt,
            "foot_uv": fuv}  # fmt: skip


def test_load_kick_measure_run(kick_folder):
    for entry in (kick_folder["run"], kick_folder["threed"], kick_folder["predict"]):
        run = km.load_kick_measure_run(entry)
        assert run.run_dir == kick_folder["run"].resolve()
        assert run.fps == FPS and run.n_frames == N_FRAMES
        assert (run.width, run.height) == (W, H)
        assert run.stem == STEM
        assert run.cameras.counts()["dlt3d"] == len(kick_folder["dlt"])
        assert run.cameras.counts().get("approximate", 0) == 0
        assert len(run.ball_uv) == N_FRAMES
        assert run.ball_source.endswith("ball.csv")
    assert any("video size unknown" in w for w in run.warnings)
    # the cameras of the run measure the ball in metres of the pitch
    g = run.cameras.ground_point(4, *run.ball_uv[4], z=0.11 + 0.05 * 4)
    assert np.hypot(g[0] - (-10.0 + 4.0), g[1] + 3.0) < 0.05
    # people: the FreeKiki track wins; the others are placed with the camera of their frame
    assert run.n_people() == 2
    p3, p4 = sorted(run.people_at(0), key=lambda p: p.id)
    assert p3.foot_source == "ground_tracks" and p3.foot_xy == (1.0, 2.0)
    assert p4.foot_source == "camera" and p4.foot_xy is not None
    assert p3.foot_uv == pytest.approx(
        (kick_folder["foot_uv"][0], kick_folder["foot_uv"][1]), abs=1e-6
    )
    assert run.people_at(99) == []
    # markers clicked by hand win over the detection, when asked for or in auto mode
    (kick_folder["run"].parent / f"{STEM}_markers.csv").write_text(
        "frame,p0_x,p0_y\n2,100,200\n", encoding="utf-8"
    )
    hand = km.load_kick_measure_run(kick_folder["run"])
    assert hand.ball_uv == {2: (100.0, 200.0)} and hand.ball_source.endswith("_markers.csv")
    assert (
        len(km.load_kick_measure_run(kick_folder["run"], ball_source="detect").ball_uv) == N_FRAMES
    )
    assert km.load_kick_measure_run(kick_folder["run"], with_people=False).people == {}


def test_load_rejects_what_is_not_a_kick_folder(tmp_path):
    with pytest.raises(ValueError, match="kick_analysis"):
        km.load_kick_measure_run(tmp_path)
    with pytest.raises(ValueError):
        km.load_kick_measure_run(tmp_path / "missing")


def test_cli_info_and_export(kick_folder, tmp_path, capsys):
    assert km.main(["--run", str(kick_folder["run"]), "--info"]) == 0
    out = capsys.readouterr().out
    assert (
        "10 DLT3D (kept), 20 solved (fixed centre), 0 tracked (frame to frame), 0 approximate"
        in out
    )
    assert "ball     : 30 frames" in out and "people   : 2 ids" in out
    target = tmp_path / "out" / "cameras.json"
    assert km.main(["--run", str(kick_folder["run"]), "--export", str(target)]) == 0
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["cameras"] == {"dlt3d": 10, "solved": 20}
    assert len(data["camera_track"]["frames"]) == N_FRAMES
    f0 = data["camera_track"]["frames"][0]
    assert f0["source"] == "dlt3d" and len(f0["matrix"]) == 12
    assert km.main(["--run", str(tmp_path)]) == 2
    assert "kick_analysis" in capsys.readouterr().err


def test_write_cameras_and_the_loader_prefers_the_saved_file(kick_folder, capsys):
    run_dir = kick_folder["run"]
    # no SAM 3 output folder yet: the CLI says so and fails
    assert km.main(["--run", str(run_dir), "--write-cameras"]) == 2
    assert "no SAM 3 output folder" in capsys.readouterr().err
    sam = run_dir / "work" / STEM / "sam3dinov3" / f"processed_sam3dinov3_{TS}" / STEM
    sam.mkdir(parents=True)
    assert km.main(["--run", str(run_dir), "--write-cameras"]) == 0
    out = capsys.readouterr().out
    assert f"{STEM}_frames.dlt3d" in out and "30 rows; 10 dlt3d, 20 solved" in out
    for name in ("_frames.dlt3d", "_frames.dlt2d", "_frames_cameras.csv", "_frames_cameras.json"):
        assert (sam / f"{STEM}{name}").is_file()
    assert km.find_frame_calibration(run_dir / "work" / STEM, STEM) == sam / f"{STEM}_frames.dlt3d"
    fresh = km.load_kick_measure_run(run_dir, use_saved=False)
    saved = km.load_kick_measure_run(run_dir)
    assert any("cameras read from" in w for w in saved.warnings)
    assert not any("cameras read from" in w for w in fresh.warnings)
    assert saved.cameras.counts() == fresh.cameras.counts()
    for f in (0, 4, 29):
        assert saved.cameras.project(f, np.array([[3.0, 2.0, 0.0]])) == pytest.approx(
            fresh.cameras.project(f, np.array([[3.0, 2.0, 0.0]])), abs=1e-6
        )
    # --no-track ignores the saved file as well
    assert not any(
        "cameras read from" in w
        for w in km.load_kick_measure_run(run_dir, use_saved=False).warnings
    )
    # a saved file older than the detect run is not trusted
    import os

    old = sam / f"{STEM}_frames.dlt3d"
    os.utime(old, (1_000_000, 1_000_000))
    assert not any("cameras read from" in w for w in km.load_kick_measure_run(run_dir).warnings)
    # a file with another frame count is rebuilt, with a warning
    os.utime(old, None)
    lines = old.read_text(encoding="utf-8").splitlines()
    old.write_text("\n".join(lines[:-3]) + "\n", encoding="utf-8")
    again = km.load_kick_measure_run(run_dir)
    assert any("rebuilt" in w for w in again.warnings) and again.n_frames == N_FRAMES


# ----------------------------------------------------------------------------------------------
# 4. People, 2D poses, pitch lines, metrics, scene export (Kick Measure panel)
# ----------------------------------------------------------------------------------------------

KPT = {"nose": 0, "left-hip": 9, "right-hip": 10, "left-ankle": 13, "right-ankle": 14,
       "right-wrist": 41, "left-wrist": 62, "neck": 69}  # fmt: skip
KPT_NAMES = [next((k for k, v in KPT.items() if v == i), f"kpt{i}") for i in range(70)]
SPEED = 5.0  # m/s of person 3 along x
BALL_XY = (20.0, 0.0)


def _pelvis_z(f: int) -> float:
    """Standing at 0.9 m, a 0.4 m jump peaking at frame 15."""
    return 0.9 + 0.4 * max(0.0, 1.0 - abs(f - 15) / 5.0)


def _x3(f: int) -> float:
    return SPEED * f / FPS


@pytest.fixture
def scene_run(kick_folder):
    """The synthetic kick folder with people, tracks, 3D / 2D poses, ball, teams and a referee."""
    run_dir, threed = kick_folder["run"], kick_folder["threed"]
    pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4],
            "role": ["kicker", "other", "keeper", "other"],
            "camera": [threed.name] * 4,
            "sam_id": [3, 4, 5, 6],
        }
    ).to_csv(threed / "player_map.csv", index=False)
    (threed / "data").mkdir()
    pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4],
            "role": ["kicker", "other", "other", "referee_main"],
            "team": ["TEAM_A", "TEAM_B", "", "REF"],
        }
    ).to_csv(threed / "data" / "team_map.csv", index=False)
    (threed / "referees").mkdir()
    (threed / "referees" / "P4_referee_main.c3d").write_bytes(b"")  # P<player_id> -> sam id 6
    sam = run_dir / "work" / STEM / "sam3dinov3" / "p1" / STEM / "sam3" / "sam_tracks.csv"
    rows = [(f, 3, 300, 400) for f in range(30)] + [(f, 4, 500, 400) for f in range(10)]
    rows += [(f, 5, 700, 400) for f in range(30)] + [(f, 6, 900, 400) for f in range(5)]
    pd.DataFrame(rows, columns=pd.Index(["frame", "obj_id", "x_px", "y_px"])).assign(
        w_px=40.0, h_px=90.0
    ).to_csv(sam, index=False)
    # ground tracks: person 3 runs at 5 m/s along x, person 4 stands at (10, 0)
    gt = [(f, 3, _x3(f), 0.0) for f in range(30)] + [(f, 4, 10.0, 0.0) for f in range(10)]
    pd.DataFrame(gt, columns=pd.Index(["frame", "person_id", "x_m", "y_m"])).to_csv(
        threed / "ground_tracks.csv", index=False
    )
    ball = [
        {"frame": f, "cx": 0.0, "cy": 0.0, "score": 1.0, "x_m": BALL_XY[0] if f >= 5 else np.nan,
         "y_m": BALL_XY[1] if f >= 5 else np.nan, "z_m": 0.11, "on_field": f >= 5}
        for f in range(30)
    ]  # fmt: skip
    pd.DataFrame(ball).to_csv(threed / "ball_3d.csv", index=False)
    poses = []
    for f in range(30):
        for i in range(70):
            # person 3: moves at 5 m/s; its ground row (kpt 999) is deliberately 100 m away
            z3 = _pelvis_z(f) if i in (9, 10) else (0.1 if i in (13, 14) else 1.0 + 0.001 * i)
            poses.append((f, 3, i, KPT_NAMES[i], _x3(f), 0.01 * i, z3))
            # person 5 (no ground track): stands at (-10, 5), wrists at 1.5 m
            z5 = 0.1 if i in (13, 14) else (1.5 if i in (41, 62) else 1.0)
            poses.append((f, 5, i, KPT_NAMES[i], -10.0, 5.0, z5))
        poses.append((f, 3, 999, "ground", _x3(f) + 100.0, 0.0, 0.0))
    pd.DataFrame(
        poses, columns=pd.Index(["frame", "person_id", "kpt_idx", "kpt_name", "x_m", "y_m", "z_m"])
    ).to_csv(threed / "poses_3d_on_field.csv", index=False)
    out2d = run_dir / "work" / STEM / "sam3dinov3" / "processed_sam3dinov3_20261123_100001" / STEM
    out2d.mkdir(parents=True)
    for pid in (3, 4):
        cols = {"frame": np.arange(30)}
        for i in range(70):
            cols[f"p{i}_x"] = 100.0 * pid + np.arange(30) + i
            cols[f"p{i}_y"] = 50.0 * pid + i + 0.5
        df = pd.DataFrame(cols)
        df.loc[1, df.columns[1:]] = np.nan  # frame 1: nothing detected
        df.loc[2, ["p5_x", "p5_y"]] = np.nan  # frame 2: one point missing
        df.to_csv(out2d / f"{STEM}_id_{pid:02d}_markers.csv", index=False)
    return km.load_kick_measure_run(run_dir)


def test_load_people_info_roles_teams_referee(scene_run):
    info = km.load_people_info(scene_run)
    assert sorted(info) == [3, 4, 5, 6]
    assert (info[3].role, info[3].team) == ("kicker", "A")
    assert (info[4].role, info[4].team) == ("other", "B")
    assert (info[5].role, info[5].team) == ("keeper", "")
    # P4_referee_main.c3d names player_id 4 -> SAM id 6 through player_map (not SAM id 4)
    assert (info[6].role, info[6].team) == ("referee", "")
    assert (info[3].first_frame, info[3].last_frame, info[3].n_frames) == (0, 29, 30)
    assert (info[4].first_frame, info[4].last_frame, info[4].n_frames) == (0, 9, 10)
    assert (info[3].has_pose_3d, info[4].has_pose_3d, info[5].has_pose_3d) == (True, False, True)
    assert (info[3].has_pose_2d, info[4].has_pose_2d, info[5].has_pose_2d) == (True, True, False)
    assert all(isinstance(p, km.PersonInfo) for p in info.values())


def test_load_people_info_missing_files(kick_folder, scene_run):
    # a bare run (no player_map, team_map, poses, referees): empty answers, no crash
    bare = km.load_kick_measure_run(kick_folder["run"])
    for f in ("player_map.csv", "poses_3d_on_field.csv"):
        (bare.threed / f).unlink(missing_ok=True)
    info = km.load_people_info(bare)
    # without player_map a referee file number is read as a SAM id (P4 -> 4)
    assert {i for i, p in info.items() if p.role == "referee"} == {4}
    assert info[3].role == "other" and info[5].role == "other"
    for f in ("referees/P4_referee_main.c3d", "data/team_map.csv"):
        (bare.threed / f).unlink()
    info = km.load_people_info(bare)
    assert {p.role for p in info.values()} == {"other"}
    assert {p.team for p in info.values()} == {""}
    assert not any(p.has_pose_3d for p in info.values())


def test_load_pose_2d_and_connections(scene_run):
    poses = km.load_pose_2d(scene_run)
    assert {i for _, i in poses} == {3, 4}
    assert (1, 3) not in poses  # nothing detected in frame 1
    xy = poses[(5, 3)]
    assert xy.shape == (70, 2)
    assert xy[0] == pytest.approx([300.0 + 5, 150.5])
    assert xy[7] == pytest.approx([300.0 + 5 + 7, 150.5 + 7])
    assert np.isnan(poses[(2, 3)][5]).all() and np.isfinite(poses[(2, 3)][6]).all()
    conn = km.pose_connections()
    assert len(conn) == 48 and all(0 <= a < 70 and 0 <= b < 70 for a, b in conn)
    assert (69, 0) in conn  # the first bone of sam3dinov3_mhr70_body.json
    # no processed folder: empty
    for f in (scene_run.run_dir / "work" / STEM / "sam3dinov3").glob("processed_*/*/*_markers.csv"):
        f.unlink()
    assert km.load_pose_2d(scene_run) == {}


def test_field_segments_px_match_the_projection(scene_run):
    tmpl = km.load_field_template()
    for f in (0, 4):  # a DLT3D frame and a solved one
        segs = km.field_segments_px(scene_run, f)
        assert len(segs) == 49 + 48  # kiki49 lines + the centre circle in 48 chords
        # the first line is p0 -> p13 (top-left corner to the penalty box)
        uv = scene_run.cameras.project(f, tmpl[[0, 13]])
        assert segs[0][0] == pytest.approx(tuple(uv[0]), abs=1e-6)
        assert segs[0][1] == pytest.approx(tuple(uv[1]), abs=1e-6)
        # the chords of the circle close on the centre-spot radius (9.15 m): project two chords back
        a, b = segs[-1]
        ga = scene_run.cameras.ground_point(f, *a)
        gb = scene_run.cameras.ground_point(f, *b)
        assert np.hypot(*ga[:2]) == pytest.approx(9.15, abs=1e-3)
        assert np.hypot(*gb[:2]) == pytest.approx(9.15, abs=1e-3)
    # a camera looking away from the pitch sees no line: every end is behind it
    K = np.array([[2200.0, 0, W / 2], [0, 2200.0, H / 2], [0, 0, 1.0]])
    zc = np.array([0.0, -1.0, 0.0])  # the true camera looks towards +y; this one towards -y
    xc = np.cross(zc, [0.0, 0.0, 1.0])
    xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    Rm = np.array([xc, yc, zc])
    P = K @ np.c_[Rm, -Rm @ CENTRE]
    looking_away = dataclasses.replace(
        scene_run, cameras=km.CameraTrack.from_matrices({0: P}, n_frames=N_FRAMES)
    )
    assert km.field_segments_px(looking_away, 0) == []


def test_ground_and_ball_series_sources(scene_run):
    g3 = km.person_ground_series(scene_run, 3)
    assert g3.shape == (N_FRAMES, 2)
    assert g3[7] == pytest.approx([_x3(7), 0.0])  # ground_tracks wins over the pose ground row
    g5 = km.person_ground_series(scene_run, 5)  # no track: mean of the ankles
    assert np.allclose(g5, [-10.0, 5.0])
    g6 = km.person_ground_series(scene_run, 6)  # nothing but a SAM box: the camera sees its foot
    assert np.isfinite(g6[:5]).all() and np.isnan(g6[5:]).all()
    assert np.isnan(km.person_ground_series(scene_run, 99)).all()
    ball = km.ball_ground_series(scene_run)
    assert np.isnan(ball[:5]).all() and np.allclose(ball[5:], BALL_XY)
    (scene_run.threed / "ball_3d.csv").unlink()
    assert np.isnan(km.ball_ground_series(scene_run)).all()


def test_body_point_series(scene_run):
    pelvis = km.body_point_series(scene_run, 3, "pelvis")
    assert pelvis.shape == (N_FRAMES, 3)
    assert pelvis[15] == pytest.approx([_x3(15), 0.01 * 9.5, 1.3])
    head = km.body_point_series(scene_run, 3, "head")
    assert head[4] == pytest.approx([_x3(4), 0.0, 1.0])
    knee = km.body_point_series(scene_run, 3, "kpt20")  # any keypoint name of the file
    assert knee[0, 1] == pytest.approx(0.2)
    with pytest.raises(ValueError, match="no 3D pose"):
        km.body_point_series(scene_run, 4, "pelvis")
    with pytest.raises(ValueError, match="unknown body point"):
        km.body_point_series(scene_run, 3, "elbow_of_doom")


def test_metric_speed_of_a_foot_at_five_metres_per_second(scene_run):
    m = km.metric_series(scene_run, "speed", 3)
    assert (m.kind, m.person, m.unit) == ("speed", 3, "m/s")
    assert np.allclose(m.values, SPEED, atol=1e-6)
    s = m.summary()
    assert s["max"] == pytest.approx(SPEED) and s["min"] == pytest.approx(SPEED)
    assert s["n_valid"] == N_FRAMES and s["unit"] == "m/s" and s["label"] == m.label
    assert m.at(7) == pytest.approx(SPEED) and math.isnan(m.at(-1)) and math.isnan(m.at(999))
    assert km.metric_series(scene_run, "speed", 3, smooth_window=0).at(10) == pytest.approx(SPEED)


def test_metric_distances_height_jump_and_point_speed(scene_run):
    d = km.metric_series(scene_run, "distance_ball", 3)
    assert math.isnan(d.at(3)) and d.at(6) == pytest.approx(BALL_XY[0] - _x3(6))  # ball on pitch
    assert d.unit == "m"
    d5 = km.metric_series(scene_run, "distance_ball", 5)
    assert d5.at(10) == pytest.approx(math.hypot(30.0, 5.0))
    dp = km.metric_series(scene_run, "distance_person", 3, other=4)
    assert dp.other == 4 and dp.at(3) == pytest.approx(10.0 - _x3(3))
    assert math.isnan(dp.at(12))  # person 4 only exists up to frame 9
    h = km.metric_series(scene_run, "height_kpt", 5, point="right_wrist")
    assert h.point == "right_wrist" and np.allclose(h.values, 1.5) and h.unit == "m"
    j = km.metric_series(scene_run, "jump", 3)
    assert j.summary()["max"] == pytest.approx(0.4) and j.summary()["frame_of_max"] == 15
    assert j.at(0) == pytest.approx(0.0) and j.at(25) == pytest.approx(0.0)
    ks = km.metric_series(scene_run, "kpt_speed", 3, point="right_ankle")
    assert np.allclose(ks.values, SPEED, atol=1e-6) and ks.unit == "m/s"
    assert ks.point == "right_ankle"
    assert set(km.METRIC_KINDS) == {"speed", "distance_ball", "distance_person", "height_kpt",
                                    "jump", "kpt_speed"}  # fmt: skip
    assert {"pelvis", "head", "neck", "left_wrist", "right_wrist", "left_ankle",
            "right_ankle"} == set(km.BODY_POINTS)  # fmt: skip


def test_metric_series_errors_are_readable(scene_run):
    with pytest.raises(ValueError, match="unknown metric"):
        km.metric_series(scene_run, "teleport", 3)
    with pytest.raises(ValueError, match="second person"):
        km.metric_series(scene_run, "distance_person", 3)
    with pytest.raises(ValueError, match="two different people"):
        km.metric_series(scene_run, "distance_person", 3, other=3)
    with pytest.raises(ValueError, match="person 99 has no position"):
        km.metric_series(scene_run, "speed", 99)
    with pytest.raises(ValueError, match="person 99 has no position"):
        km.metric_series(scene_run, "distance_person", 3, other=99)
    with pytest.raises(ValueError, match="no 3D pose"):
        km.metric_series(scene_run, "height_kpt", 4, point="head")
    with pytest.raises(ValueError, match="no 3D pose"):
        km.metric_series(scene_run, "jump", 4)
    with pytest.raises(ValueError, match="unknown body point"):
        km.metric_series(scene_run, "kpt_speed", 3, point="tail")
    with pytest.raises(ValueError, match="needs a body point"):
        km.metric_series(scene_run, "height_kpt", 3, point=None)
    (scene_run.threed / "ball_3d.csv").unlink()
    with pytest.raises(ValueError, match="no ball"):
        km.metric_series(scene_run, "distance_ball", 3)
    (scene_run.threed / "poses_3d_on_field.csv").unlink()
    with pytest.raises(ValueError, match="no 3D pose"):
        km.metric_series(scene_run, "height_kpt", 3, point="head")


def _tree(folder: Path) -> dict[str, tuple[int, int]]:
    return {
        str(p.relative_to(folder)): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in sorted(folder.rglob("*"))
        if p.is_file()
    }


def test_export_scene_files(scene_run, tmp_path):
    import ezc3d

    before = _tree(scene_run.run_dir)
    out = tmp_path / "scene"
    metrics = [
        km.metric_series(scene_run, "speed", 3),
        km.metric_series(scene_run, "jump", 3),
        SimpleNamespace(kind="height_kpt", person=5, other=None, point="right_wrist"),
        SimpleNamespace(kind="height_kpt", person=4, other=None, point="head"),  # fails: no pose
    ]
    files = km.export_scene(scene_run, out, [3, 4, 5], metrics=metrics)
    assert _tree(scene_run.run_dir) == before  # the input run is never touched
    assert set(files) == {"c3d", "c3d_csv", "bvh_3", "bvh_5", "ball_json", "pitch_json",
                          "blender_script", "metrics_csv", "metrics_json", "readme"}  # fmt: skip
    assert all(p.is_file() and out in p.parents for p in files.values())

    c3d = ezc3d.c3d(str(files["c3d"]))
    labels = list(c3d["parameters"]["POINT"]["LABELS"]["value"])
    assert len(labels) == 2 * 70 + 1 + 49
    assert labels[0] == "P3_nose" and "P5_right-wrist" in labels and "BALL" in labels
    assert "FKP_center_field" in labels and "FKP_top_left_corner" in labels
    assert "P4_nose" not in labels  # person 4 has no 3D pose
    pts = c3d["data"]["points"]
    assert pts.shape[2] == N_FRAMES and c3d["parameters"]["POINT"]["RATE"]["value"][0] == FPS
    i = labels.index("P3_left-hip")
    assert pts[0, i, 10] == pytest.approx(_x3(10), abs=1e-3)
    assert pts[2, i, 15] == pytest.approx(1.3, abs=1e-3)
    ball = labels.index("BALL")
    assert np.isnan(pts[0, ball, 2]) and pts[0, ball, 7] == pytest.approx(BALL_XY[0], abs=1e-3)
    c = labels.index("FKP_center_field")
    assert np.allclose(pts[:3, c, 3], 0.0, atol=1e-6)  # the pitch template is static

    header = files["c3d_csv"].read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header[:2] == ["frame", "time_s"] and "P3_nose_X" in header and "BALL_Z" in header
    assert len(files["c3d_csv"].read_text(encoding="utf-8").splitlines()) == N_FRAMES + 1

    for pid in (3, 5):
        bvh = files[f"bvh_{pid}"]
        assert bvh.parent.name == "blender" and bvh.name == f"person_{pid}.bvh"
        text = bvh.read_text(encoding="utf-8")
        assert f"Frames: {N_FRAMES}" in text and "ROOT nose" in text and "MOTION" in text
    bj = json.loads(files["ball_json"].read_text(encoding="utf-8"))
    assert bj["n_frames"] == N_FRAMES and bj["xyz"][0] is None and bj["xyz"][6] == [20.0, 0.0, 0.11]
    pj = json.loads(files["pitch_json"].read_text(encoding="utf-8"))
    assert len(pj["segments"]) == 97

    script = files["blender_script"].read_text(encoding="utf-8")
    compile(script, "kick_scene_blender.py", "exec")  # plain bpy code: it must at least parse
    assert "import bpy" in script and "person_3.bvh" in script and "person_5.bvh" in script
    assert script.index("scene.frame_end") > script.index("_import_people()")  # rate set LAST
    assert "bpy" not in sys.modules  # vaila never imports bpy

    mcsv = pd.read_csv(files["metrics_csv"])
    assert list(mcsv.columns[:2]) == ["frame", "time_s"] and len(mcsv) == N_FRAMES
    assert "speed P3 (m/s)" in mcsv.columns and "jump P3 (m)" in mcsv.columns
    assert "height right wrist P5 (m)" in mcsv.columns
    assert mcsv["speed P3 (m/s)"].iloc[10] == pytest.approx(SPEED, abs=1e-4)
    summary = json.loads(
        files["metrics_json"].read_text(encoding="utf-8"),
        parse_constant=lambda c: pytest.fail(f"non-strict JSON constant {c}"),
    )
    assert [m["label"] for m in summary["metrics"]] == [
        "speed P3", "jump P3", "height right wrist P5"]  # fmt: skip
    assert (
        summary["metrics"][1]["max"] == pytest.approx(0.4)
        and summary["metrics"][1]["frame_of_max"] == 15
    )
    assert len(summary["errors"]) == 1 and "P4" in summary["errors"][0]
    readme = files["readme"].read_text(encoding="utf-8")
    assert "kick_scene.c3d" in readme and "kick_scene_blender.py" in readme
    assert "P3 (kicker, team A)" in readme and "P4" in readme  # P4 is listed as skipped


def test_export_scene_options_and_ball_override(scene_run, tmp_path):
    only_metrics = km.export_scene(
        scene_run, tmp_path / "m", [3], metrics=[km.metric_series(scene_run, "speed", 3)],
        mkvis3d=False, blender=False,
    )  # fmt: skip
    assert set(only_metrics) == {"metrics_csv", "metrics_json", "readme"}
    only_c3d = km.export_scene(scene_run, tmp_path / "c", [3], blender=False)
    assert set(only_c3d) == {"c3d", "c3d_csv", "readme"}
    assert not (tmp_path / "c" / "blender").exists()
    only_blender = km.export_scene(scene_run, tmp_path / "b", [5], mkvis3d=False)
    assert set(only_blender) == {"bvh_5", "ball_json", "pitch_json", "blender_script", "readme"}
    # a ball given by the caller (frame -> x, y, z) replaces ball_3d.csv
    given = {2: (1.0, 2.0, 3.0), 3: (1.5, 2.0, 3.5)}
    files = km.export_scene(scene_run, tmp_path / "g", [3], ball_xyz=given, blender=True)
    bj = json.loads(files["ball_json"].read_text(encoding="utf-8"))
    assert bj["xyz"][2] == [1.0, 2.0, 3.0] and bj["xyz"][6] is None
    import ezc3d

    labels = list(ezc3d.c3d(str(files["c3d"]))["parameters"]["POINT"]["LABELS"]["value"])
    pts = ezc3d.c3d(str(files["c3d"]))["data"]["points"]
    assert pts[2, labels.index("BALL"), 3] == pytest.approx(3.5, abs=1e-3)
    # nothing to write: a person without 3D pose and no ball at all
    (scene_run.threed / "ball_3d.csv").unlink()
    with pytest.raises(ValueError, match="nothing to export"):
        km.export_scene(scene_run, tmp_path / "none", [4])


def test_find_mkvis3d_env_and_argv(tmp_path, monkeypatch):
    fake = tmp_path / "viewer" / "mkvis3d.py"
    fake.parent.mkdir()
    fake.write_text("print('hi')\n", encoding="utf-8")
    monkeypatch.setenv("VAILA_MKVIS3D", str(fake))
    assert km.find_mkvis3d() == fake
    assert km.mkvis3d_argv("scene.c3d") == [sys.executable, str(fake), "scene.c3d"]
    # an env path that does not exist falls back to <repo root>/../mkvis3d/mkvis3d.py
    monkeypatch.setenv("VAILA_MKVIS3D", str(tmp_path / "missing.py"))
    repo = tmp_path / "work" / "vaila"
    repo.mkdir(parents=True)
    monkeypatch.setattr(km, "_repo_root", lambda: repo)
    assert km.find_mkvis3d() is None and km.mkvis3d_argv("scene.c3d") is None
    sibling = tmp_path / "work" / "mkvis3d" / "mkvis3d.py"
    sibling.parent.mkdir()
    sibling.write_text("", encoding="utf-8")
    assert km.find_mkvis3d() == sibling


def test_cli_export_scene(scene_run, tmp_path, capsys):
    out = tmp_path / "cli_scene"
    argv = ["--run", str(scene_run.run_dir), "--export-scene", str(out), "--people", "3,5",
            "--metric", "speed:3", "--metric", "distance_person:3:4",
            "--metric", "height_kpt:5:right_wrist", "--no-blender"]  # fmt: skip
    assert km.main(argv) == 0
    text = capsys.readouterr().out
    assert "speed P3" in text and "distance P3-P4" in text and "height right wrist P5" in text
    assert (out / "kick_scene.c3d").is_file() and (out / "metrics.csv").is_file()
    assert not (out / "blender").exists()
    # without --people: every person with a 3D pose; --no-mkvis3d drops the C3D
    out2 = tmp_path / "cli_all"
    assert km.main(["--run", str(scene_run.run_dir), "--export-scene", str(out2),
                    "--no-mkvis3d"]) == 0  # fmt: skip
    capsys.readouterr()
    assert (out2 / "blender" / "person_3.bvh").is_file() and (
        out2 / "blender" / "person_5.bvh"
    ).is_file()
    assert not (out2 / "kick_scene.c3d").exists()
    # errors are readable and return 2
    for bad, needle in (
        (["--metric", "speed"], "KIND:PERSON"),
        (["--metric", "speed:x"], "must be a number"),
        (["--metric", "distance_person:3"], "needs the other person"),
        (["--metric", "jump:4"], "no 3D pose"),
        (["--people", "a,b"], "--people"),
    ):
        assert km.main(["--run", str(scene_run.run_dir), "--export-scene", str(tmp_path / "x"),
                        *bad]) == 2  # fmt: skip
        assert needle in capsys.readouterr().err


def test_ground_source_is_the_one_with_most_valid_frames(scene_run):
    # tie (30 vs 30 frames): ground_tracks wins, as the first of the order
    _, src = km.person_ground_source(scene_run, 3)
    assert (
        src == "ground_tracks" and km.metric_series(scene_run, "speed", 3).source == "ground_tracks"
    )
    assert km.person_ground_source(scene_run, 5)[1] == "ankles"  # ankles 30 vs no box
    assert km.person_ground_source(scene_run, 6)[1] == "box_foot"
    assert km.person_ground_source(scene_run, 99)[1] == ""

    # ground_tracks covers 5 frames, the SAM box of person 3 walks at 5 m/s over 30 frames
    truth = _calibration()[0]
    sam = next((scene_run.run_dir / "work" / STEM / "sam3dinov3").glob("p1/*/sam3/sam_tracks.csv"))
    rows = []
    for f in range(N_FRAMES):
        u, v = km.project_points(truth[f], np.array([_x3(f) - 10.0, -4.0, 0.0]))[0]
        rows.append((f, 3, u - 20.0, v - 90.0, 40.0, 90.0))
    pd.DataFrame(
        rows, columns=pd.Index(["frame", "obj_id", "x_px", "y_px", "w_px", "h_px"])
    ).to_csv(sam, index=False)
    gt = pd.read_csv(scene_run.threed / "ground_tracks.csv")
    gt[~((gt.person_id == 3) & (gt.frame >= 5))].to_csv(
        scene_run.threed / "ground_tracks.csv", index=False
    )
    # the pose ground row (100 m away) and the ankles also have 30 frames: remove them from the pose file
    poses = pd.read_csv(scene_run.threed / "poses_3d_on_field.csv")
    poses[poses.person_id != 3].to_csv(scene_run.threed / "poses_3d_on_field.csv", index=False)
    run = km.load_kick_measure_run(scene_run.run_dir)
    xy, src = km.person_ground_source(run, 3)
    assert src == "box_foot" and np.isfinite(xy).all()  # 30 box frames beat 5 track frames
    assert np.hypot(*(xy[10] - [_x3(10) - 10.0, -4.0])) < 0.1  # the box foot seen by the camera
    m = km.metric_series(run, "speed", 3)
    assert m.source == "box_foot" and m.summary()["source"] == "box_foot"
    assert m.summary()["n_valid"] == N_FRAMES  # valid on every box frame, not only on the 5 tracks
    assert np.allclose(m.values[2:-2], SPEED, atol=0.8)
    # a distance names both sources
    d = km.metric_series(run, "distance_person", 3, other=4)
    assert d.source == "box_foot+ground_tracks"


def test_box_foot_skips_approximate_cameras_and_smooths_wider(scene_run):
    truth = _calibration()[0]
    # exact cameras on every frame except 12-17 (copies of a neighbour = "approximate")
    mats = {f: truth[f] for f in range(N_FRAMES) if not 12 <= f <= 17}
    track = km.CameraTrack.from_matrices(mats, n_frames=N_FRAMES)
    assert all(track.is_approximate(f) for f in range(12, 18))
    sam = next((scene_run.run_dir / "work" / STEM / "sam3dinov3").glob("p1/*/sam3/sam_tracks.csv"))
    rows = []
    for f in range(N_FRAMES):
        u, v = km.project_points(truth[f], np.array([_x3(f) - 10.0, -4.0, 0.0]))[0]
        rows.append((f, 3, u - 20.0, v - 90.0, 40.0, 90.0))
    pd.DataFrame(
        rows, columns=pd.Index(["frame", "obj_id", "x_px", "y_px", "w_px", "h_px"])
    ).to_csv(sam, index=False)
    gt = pd.read_csv(scene_run.threed / "ground_tracks.csv")
    gt[~((gt.person_id == 3) & (gt.frame >= 5))].to_csv(
        scene_run.threed / "ground_tracks.csv", index=False
    )
    poses = pd.read_csv(scene_run.threed / "poses_3d_on_field.csv")
    poses[poses.person_id != 3].to_csv(scene_run.threed / "poses_3d_on_field.csv", index=False)
    run = dataclasses.replace(km.load_kick_measure_run(scene_run.run_dir), cameras=track)
    xy, src = km.person_ground_source(run, 3)
    assert src == "box_foot"
    good = [f for f in range(N_FRAMES) if not 12 <= f <= 17]
    assert np.isfinite(xy[good]).all() and np.isnan(xy[12:18]).all()
    m = km.metric_series(run, "speed", 3)
    assert m.source == "box_foot" and m.summary()["n_valid"] == len(good)
    assert np.allclose(m.values[good], SPEED, atol=0.05)
    # a noisy box bottom (+-1 m of grass, alternating) is tamed by the wider window
    noisy = xy.copy()
    noisy[::2, 0] += 0.5
    noisy[1::2, 0] -= 0.5
    w5 = km.speed_series(noisy, FPS, 5)
    w9 = km.speed_series(noisy, FPS, 9)
    assert np.nanmax(w9) < np.nanmax(w5)
