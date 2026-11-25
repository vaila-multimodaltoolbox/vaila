"""Kick contact from the feet of the people (open play: the ball is dribbled before the kick).

Synthetic scenes only: no GPU, no model, no video. ``contact_from_feet`` works on image pixels (the
ball and the feet keypoints of SAM 3D Body), so the first group builds tracks in pixels; the second
group simulates a dribble then a physical kick, seen by a synthetic camera, and runs the automatic
flight, the camera helpers and ``write_field_3d`` through it.

Version: 0.4.19
Update Date: 23 November 2026
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from vaila import freekiki_auto_flight as af
from vaila import freekiki_detect3d as d3
from vaila.fifa_to_dlt import compute_dlt3d_from_KRt
from vaila.freekiki_field3d import ball_3d
from vaila.freekiki_physics import BALL_RADIUS, simulate_flight
from vaila.monocular_dlt_align import dlt_projection_matrix

FPS = 30.0
BOX_H = 130.0

# --- pixel scenes: a player dribbles, kicks, and others stand around ---------------------------------


def _kick_scene(kick: int = 60, n: int = 100, push: tuple[float, float] = (50.0, -8.0)):
    """Player 22 runs right at 3 px/frame with the ball at his foot (it wanders 14..38 px ahead),
    kicks it at ``kick`` (the ball leaves at ``push`` px/frame). Returns ball, feet, heights."""
    ball: dict[int, tuple[float, float]] = {}
    feet: dict[int, dict[int, list[tuple[float, float]]]] = {}
    heights: dict[int, dict[int, float]] = {}
    for f in range(n):
        fx, fy = 300.0 + 3.0 * f, 600.0
        feet[f] = {22: [(fx, fy), (fx + 12, fy + 4), (fx - 10, fy + 3)]}
        heights[f] = {22: BOX_H}
        if f <= kick:
            lead = 14.0 if f == kick else 14.0 + 12.0 * (1.0 + np.sin(0.7 * f))
            ball[f] = (fx + lead, fy - 10.0)
        else:
            bx, by = ball[kick]
            ball[f] = (bx + push[0] * (f - kick), by + push[1] * (f - kick))
    return ball, feet, heights


def _stand(feet, heights, pid: int, at: tuple[float, float], frames) -> None:
    for f in frames:
        feet.setdefault(f, {})[pid] = [at, (at[0] + 10, at[1] + 4)]
        heights.setdefault(f, {})[pid] = BOX_H


def test_a_dribbled_ball_is_kicked_at_the_last_frame_at_the_foot() -> None:
    ball, feet, heights = _kick_scene(kick=60)
    for kicker in (22, None):
        got = af.contact_from_feet(ball, feet, heights, FPS, kicker=kicker)
        assert got is not None
        assert (got["frame"], got["person"], got["source"]) == (60, 22, "feet")
        assert got["distance_px"] < 0.2 * BOX_H
        assert got["dv_h_per_s"] > 3 * af.FEET_KICK_DV_H_S  # far above the dribble touches
        assert got["others"] == []  # the dribble touches before are not kicks


def test_the_dribble_touches_alone_are_not_a_kick() -> None:
    ball, feet, heights = _kick_scene(kick=60)
    dribble = {f: uv for f, uv in ball.items() if f <= 60}  # the clip ends at the kick
    assert af.contact_from_feet(dribble, feet, heights, FPS, kicker=22) is None
    slow = _kick_scene(kick=60, push=(5.0, -1.0))  # a nudge, not a kick
    assert af.contact_from_feet(slow[0], slow[1], slow[2], FPS, kicker=22) is None


def test_a_ball_passing_a_foot_at_speed_is_not_a_contact() -> None:
    ball, feet, heights = _kick_scene(kick=60)
    bx, by = ball[60]
    # a second player stands exactly on the ball's flight path: the ball passes his foot at frame 70
    path = (bx + 50.0 * 10, by - 8.0 * 10)
    _stand(feet, heights, 9, path, range(100))
    near = float(np.hypot(*(np.asarray(ball[70]) - np.asarray(path))))
    assert near < 1.0  # it really goes over his foot
    got = af.contact_from_feet(ball, feet, heights, FPS)
    assert got is not None and (got["frame"], got["person"]) == (60, 22)
    assert af.contact_from_feet(ball, feet, heights, FPS, kicker=9) is None  # no kick by him
    # the same pass-by with no kick at all anywhere
    flying = {f: (0.0 + 50.0 * f, 500.0 - 8.0 * f) for f in range(40)}
    only: dict = {}
    only_h: dict = {}
    _stand(only, only_h, 9, flying[20], range(40))
    assert af.contact_from_feet(flying, only, only_h, FPS) is None


def test_a_ball_already_flying_at_the_start_has_no_contact() -> None:
    flying = {f: (300.0 + 50.0 * f, 600.0 - 8.0 * f) for f in range(30)}
    feet: dict = {}
    heights: dict = {}
    _stand(feet, heights, 3, (300.0, 600.0), range(30))  # a foot at the ball in the first frames
    assert af.contact_from_feet(flying, feet, heights, FPS) is None
    # a clip that starts at the kick: the ball going in was not seen, so it cannot be judged
    ball, feet2, heights2 = _kick_scene(kick=0, n=40)
    assert af.contact_from_feet(ball, feet2, heights2, FPS, kicker=22) is None
    # two ball frames up to the contact (one velocity) are enough to judge it
    ball1, feet1, heights1 = _kick_scene(kick=1, n=40)
    got = af.contact_from_feet(ball1, feet1, heights1, FPS, kicker=22)
    assert got is not None and got["frame"] == 1


def test_the_strongest_change_of_the_ball_wins_not_the_last_touch() -> None:
    ball, feet, heights = _kick_scene(kick=60, n=130)
    # the keeper parries it at frame 90 (the ball leaves slowly the other way): a later touch
    bx, by = ball[90]
    _stand(feet, heights, 25, (bx - 6.0, by + 6.0), range(130))
    for f in range(91, 130):
        ball[f] = (bx + 20.0 * (f - 90), by - 3.0 * (f - 90))
    got = af.contact_from_feet(ball, feet, heights, FPS)
    assert got is not None and (got["frame"], got["person"]) == (60, 22)
    assert [o["person"] for o in got["others"]] == [25]
    assert got["others"][0]["frame"] in (90, 91)  # the slow parry leaves the foot a frame late
    assert got["others"][0]["dv_h_per_s"] < got["dv_h_per_s"]
    parry = af.contact_from_feet(ball, feet, heights, FPS, kicker=25)
    assert parry is not None and parry["frame"] in (90, 91)


def test_a_missing_ball_frame_after_the_contact_does_not_lose_it() -> None:
    ball, feet, heights = _kick_scene(kick=60)
    del ball[61]
    got = af.contact_from_feet(ball, feet, heights, FPS, kicker=22)
    assert got is not None and got["frame"] == 60
    del ball[62], ball[63]  # too long a gap after the contact: the separation is not seen
    assert af.contact_from_feet(ball, feet, heights, FPS, kicker=22) is None


def test_find_contact_uses_the_feet_and_otherwise_decides_as_before() -> None:
    ball, feet, heights = _kick_scene(kick=60)
    ground = {f: (0.5 * f, 0.0) for f in range(61)}  # a ball dribbled in a line on the pitch
    found = af.find_contact(ball, ground, FPS, feet=feet, heights_px=heights, kicker=22)
    assert (found["frame"], found["source"], found["person"]) == (60, "feet", 22)
    assert found["pre_contact"] == "rolling" and found["warning"] is None
    assert found["kick_xy"][0] == pytest.approx(30.25, abs=0.05)  # on the line, half a frame on
    assert found["incoming"]["frames"] == [57, 60]
    # a contact given by hand still wins
    given = af.find_contact(ball, ground, FPS, contact_frame=58, feet=feet, heights_px=heights)
    assert (given["frame"], given["source"]) == (58, "manual")
    # no feet: today's reading, byte for byte the same result with empty feet
    plain = af.find_contact(ball, ground, FPS)
    assert af.find_contact(ball, ground, FPS, feet={}, heights_px={}, kicker=22) == plain
    assert plain["source"] != "feet"
    # feet that see no kick (another kicker id): the old rules decide
    other = af.find_contact(ball, ground, FPS, feet=feet, heights_px=heights, kicker=5)
    assert other["source"] == plain["source"] and other["frame"] == plain["frame"]


# --- a physical dribble + kick seen by a synthetic camera ---------------------------------------------

W, H_IMG = 1920, 1080
K_INT = np.array([[2400.0, 0.0, W / 2.0], [0.0, 2400.0, H_IMG / 2.0], [0.0, 0.0, 1.0]])
N = 70
KICK_F = 30  # last frame at the foot; the kick is at 30.4 frames
TK = 30.4 / FPS


def _camera(centre, target) -> np.ndarray:
    centre, target = np.asarray(centre, float), np.asarray(target, float)
    z = target - centre
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    return compute_dlt3d_from_KRt(K_INT, R, -R @ centre)


L = _camera([10.0, -38.0, 22.0], [30.0, 0.0, 1.0])
P = dlt_projection_matrix(L)


def _pix(xyz) -> tuple[float, float]:
    h = P @ np.append(np.asarray(xyz, float), 1.0)
    return float(h[0] / h[2]), float(h[1] / h[2])


def _world_scene(sign: float = 1.0):
    """The ball rolls at 4 m/s along ``sign * x`` with the kicker's foot beside it, then is kicked at
    ``TK`` (28 m/s along the line, 3 m/s across, 9 m/s up). Returns pixels, the kick point, the
    feet and heights of the kicker (SAM 7) and of a bystander (SAM 9)."""
    x0 = sign * 12.0
    theta = [x0, 0.0, sign * 28.0, 3.0, 9.0, TK, 0.0, 0.0, 0.0]
    flight = simulate_flight(np.asarray(theta, float), np.arange(N) / FPS, BALL_RADIUS)
    positions = flight.copy()
    for f in range(N):
        if f / FPS < TK:  # still rolling into the kick
            positions[f] = (x0 - sign * 4.0 * (TK - f / FPS), 0.0, BALL_RADIUS)
    pixels = {f: _pix(positions[f]) for f in range(N)}
    feet: dict[int, dict[int, list[tuple[float, float]]]] = {}
    heights: dict[int, dict[int, float]] = {}
    for f in range(N):
        # the kicker runs at the dribble pace, a little behind the ball's line
        run = x0 - sign * 4.0 * (TK - f / FPS)
        foot = np.array([run - sign * 0.3, -0.1, 0.0])
        feet[f] = {
            7: [_pix(foot + [0.0, 0.0, 0.08]), _pix(foot + [sign * 0.15, 0.0, 0.02])],
            9: [_pix([sign * 25.0, 4.0, 0.08])],  # stands still off the line
        }
        top = _pix(foot + [0.0, 0.0, 1.8])
        base = _pix(foot)
        heights[f] = {7: abs(base[1] - top[1]), 9: abs(base[1] - top[1])}
    return pixels, positions, feet, heights, x0


def _ball_rows(pixels) -> list[dict]:
    return [
        {"frame": f, "cx": uv[0], "cy": uv[1], "score": 1.0} for f, uv in sorted(pixels.items())
    ]


def test_the_kick_of_a_dribbled_ball_starts_the_flight_at_the_foot() -> None:
    pixels, positions, feet, heights, x0 = _world_scene()
    cameras = dict.fromkeys(range(N), L)
    ball_all = ball_3d(_ball_rows(pixels), cameras)
    result = af.auto_flight(
        ball_all, pixels, cameras, [], FPS, N, feet=feet, heights_px=heights, kicker=7
    )
    assert result is not None
    assert result["contact"]["source"] == "feet" and result["contact"]["person"] == 7
    assert result["onset_frame"] == KICK_F and result["side"] == 1.0
    kick_xy = (x0, 0.0)
    assert result["start"]["xyz"][:2] == pytest.approx(kick_xy, abs=0.35)
    window = result["fit"]["kick_time_s"]
    assert KICK_F / FPS - 1e-9 <= window <= (KICK_F + 1) / FPS + 1e-9
    assert result["fit"]["rms_px"] < 3.0
    assert result["barrier"] == {}  # open play: no wall
    rows, record = af.auto_rows_and_record(result, pixels, N)
    assert record["contact_source"] == "feet" and record["contact_person"] == 7
    assert record["pre_contact"] == "rolling" and record["side"] == 1.0
    assert all(r["z_m"] == BALL_RADIUS for r in rows if r["frame"] < KICK_F)  # the dribble


def test_the_goal_attacked_follows_the_ball_after_the_kick() -> None:
    pixels, _, feet, heights, _ = _world_scene(sign=-1.0)
    cameras = dict.fromkeys(range(N), L)
    ball_all = ball_3d(_ball_rows(pixels), cameras)
    result = af.auto_flight(ball_all, pixels, cameras, [], FPS, N, feet=feet, heights_px=heights)
    assert result is not None and result["contact"]["source"] == "feet"
    assert result["side"] == -1.0  # kicked towards -x: the goal at -52.45


def test_a_kick_without_feet_is_never_read_from_them() -> None:
    pixels, _, _, _, _ = _world_scene()
    cameras = dict.fromkeys(range(N), L)
    ground = {
        int(r["frame"]): (float(r["x_m"]), float(r["y_m"]))
        for r in ball_3d(_ball_rows(pixels), cameras)
    }
    assert af.find_contact(pixels, ground, FPS)["source"] != "feet"


# --- the cameras ---------------------------------------------------------------------------------------


def test_a_camera_is_eleven_coefficients_or_a_projection_of_any_scale() -> None:
    assert af.as_dlt(L) == pytest.approx(L)
    assert af.as_dlt(P) == pytest.approx(L, rel=1e-9)
    assert af.as_dlt(-37.0 * P) == pytest.approx(L, rel=1e-9)  # scale and sign are free
    assert af.as_dlt((5.0 * P).reshape(12)) == pytest.approx(L, rel=1e-9)
    with pytest.raises(ValueError):
        af.as_dlt(np.zeros(7))
    with pytest.raises(ValueError):
        af.as_dlt(np.zeros((3, 4)))
    cams = af.as_dlt_cameras({3: P, 4: L, 5: np.full((3, 4), np.nan)})
    assert sorted(cams) == [3, 4] and cams[3] == pytest.approx(cams[4], rel=1e-9)
    assert af.as_dlt_cameras(None) == {}


def test_frames_without_a_camera_are_left_out_of_the_fit() -> None:
    pixels, _, feet, heights, x0 = _world_scene()
    cameras = {f: L for f in range(N) if not 40 <= f < 50}  # an approximate stretch, dropped
    ball_all = ball_3d(_ball_rows(pixels), cameras)
    result = af.auto_flight(
        ball_all, pixels, cameras, [], FPS, N, feet=feet, heights_px=heights, kicker=7
    )
    assert result is not None and result["onset_frame"] == KICK_F
    assert result["fit"]["rms_px"] < 3.0
    assert result["start"]["xyz"][:2] == pytest.approx((x0, 0.0), abs=0.35)


class _Info:
    def __init__(self, source: str, matrix: np.ndarray) -> None:
        self.source, self.matrix = source, matrix


class _Track:
    n_frames = 6

    def info(self, frame: int) -> _Info:
        source = "approximate" if frame in (2, 3) else ("solved" if frame % 2 else "dlt3d")
        return _Info(source, 3.0 * P)


def test_cameras_from_a_track_leave_out_the_approximate_frames() -> None:
    cams = af.cameras_from_track(_Track())
    assert sorted(cams) == [0, 1, 4, 5]
    assert cams[0] == pytest.approx(L, rel=1e-9)
    assert sorted(af.cameras_from_track(_Track(), exclude=())) == [0, 1, 2, 3, 4, 5]
    assert sorted(af.cameras_from_track(_Track(), exclude=("solved", "approximate"))) == [0, 4]


# --- the predictions file and the 3D folder ----------------------------------------------------------


def _write_predictions(path: Path, feet, heights) -> None:
    names = ["nose", "left-ankle", "left-big-toe-tip", "left-wrist"]
    frames = []
    for f in sorted(feet):
        instances = []
        for pid, pts in feet[f].items():
            foot = [list(p) for p in pts]
            kps = [[0.0, 0.0], foot[0], foot[-1], [float("nan"), float("nan")]]
            top = foot[0][1] - heights[f][pid]
            instances.append(
                {
                    "sam_obj_id": pid,
                    "person_id": pid,
                    "bbox_xyxy": [foot[0][0] - 10, top, foot[0][0] + 10, foot[0][1]],
                    "keypoints_2d_px": kps,
                }
            )
        frames.append({"frame": f, "instances": instances})
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump({"keypoint_names": names, "frames": frames}, fh)


def test_read_feet_keeps_the_foot_keypoints_and_the_box_height(tmp_path: Path) -> None:
    _, _, feet, heights, _ = _world_scene()
    path = tmp_path / "maincam_sam3dinov3_predictions.json.gz"
    _write_predictions(path, feet, heights)
    got_feet, got_heights = af.read_feet(path)
    assert sorted(got_feet) == list(range(N)) and sorted(got_feet[5]) == [7, 9]
    assert len(got_feet[5][7]) == 2  # ankle and toe; the nose and the NaN wrist are not feet
    assert got_feet[5][7][0] == pytest.approx(feet[5][7][0])
    assert got_heights[5][7] == pytest.approx(heights[5][7])


def test_write_field_3d_reads_the_contact_from_the_feet_and_uses_the_ball_cameras(
    tmp_path: Path,
) -> None:
    pixels, _, feet, heights, x0 = _world_scene()
    _write_predictions(tmp_path / "maincam_sam3dinov3_predictions.json.gz", feet, heights)
    camera_csv = tmp_path / "maincam_sam3dinov3_camera.csv"
    camera_csv.write_text("frame,person_id\n", encoding="utf-8")
    rows = _ball_rows(pixels)
    per_frame = dict.fromkeys(range(N), P)  # 3x4: the form of a kick_measure.CameraTrack

    def run(name: str, **kw) -> dict | None:
        out = tmp_path / name
        out.mkdir()
        d3.write_field_3d(
            tmp_path, out, cameras={0: L}, n_frames=N, rate_hz=FPS, camera_csv=camera_csv,
            keypoints_csv=None, kept=None, ball_rows=rows, **kw,
        )  # fmt: skip
        path = out / "ball_flight.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    record = run("feet", ball_cameras=per_frame, feet_contact=True, kicker_sam=7)
    assert record is not None
    assert (record["contact_frame"], record["contact_source"], record["contact_person"]) == (
        KICK_F, "feet", 7,
    )  # fmt: skip
    assert record["anchor_points"]["p0"]["xyz"][:2] == pytest.approx((x0, 0.0), abs=0.35)
    assert record["rms_px"] < 3.0 and record["side"] == 1.0
    # the same call without the feet: today's reading, never "feet"
    plain = run("plain", ball_cameras=per_frame)
    assert plain is None or (plain["contact_source"] != "feet" and "contact_person" not in plain)
    # the default (no ball cameras): one DLT3D, so one frame with a ball on the pitch; the flight
    # then only sees a ball that "already flies in the first frame" (today's wrong reading)
    default = run("none")
    assert default is not None and default["contact_source"] == "first_frame"
    read = af.flight_contact(tmp_path / "feet")
    assert read is not None and (read["frame"], read["person"], read["source"]) == (
        KICK_F,
        7,
        "feet",
    )
    assert read["side"] == 1.0  # the goal the ball goes to, for the set-piece rule
