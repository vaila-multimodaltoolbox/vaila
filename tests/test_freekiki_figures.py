"""The kicker's posture metrics and figures (vaila/freekiki_kicker.py, freekiki_kicker_figures.py).

The posture quantities have closed forms on synthetic geometry: two planes at 90 degrees, a hip line
at 30 degrees to the kick, a foot moving 8.4 m/s across the kick, and a leg whose knee makes all of
the foot's motion (share 1) or that only translates (share 0). The figures are checked for their
files, and the overlay is checked on a real DLT projection with a real (small) video.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from vaila import freekiki_kicker as kk
from vaila import freekiki_kicker_figures as figs
from vaila.fifa_to_dlt import compute_dlt3d_from_KRt
from vaila.freekiki_stereo import CamView

FPS = 60.0
N = 120
CONTACT = 40
KICKER = 1
K = np.array([[900.0, 0.0, 160.0], [0.0, 900.0, 120.0], [0.0, 0.0, 1.0]])


def _static(n: int, xyz) -> np.ndarray:
    return np.tile(np.array(xyz, dtype=float), (n, 1))


def _posture(points: dict, n: int, kick=(1.0, 0.0), side: str = "left") -> dict:
    foot = kk._foot_marker(points, side, n)
    return kk._posture(points, side, foot, FPS, n, np.array(kick, dtype=float), np.zeros(n))


def test_the_thigh_shank_and_shank_foot_planes_meet_at_ninety_degrees() -> None:
    n = 5
    points = {
        "left-hip": _static(n, (0, 0, 1)),
        "left-knee": _static(n, (0, 0, 0.5)),
        "left-ankle": _static(n, (0.5, 0, 0.5)),
        "left-big-toe-tip": _static(n, (0.5, 0.5, 0.5)),
    }
    post = _posture(points, n)
    assert np.allclose(post["plane_deg"], 90.0)
    assert np.allclose(post["ankle_angle_deg"], 90.0)


def test_the_hip_angle_is_the_three_point_angle_shoulder_hip_knee() -> None:
    n = 3
    points = {
        "left-shoulder": _static(n, (0, 0, 1.5)),
        "right-shoulder": _static(n, (0, 0, 1.5)),
        "left-hip": _static(n, (0, 0, 1)),
        "left-knee": _static(n, (0.5, 0, 1)),
    }
    assert np.allclose(_posture(points, n)["hip_angle_deg"], 90.0)


def test_the_pelvis_rotation_is_the_hip_line_against_the_kick_direction() -> None:
    n = 3
    points = {
        "left-hip": _static(n, (0, 0, 1)),
        "right-hip": _static(n, (0.2598, 0.15, 1)),  # the hip line at 30 degrees to +x
    }
    assert np.allclose(_posture(points, n)["pelvis_deg"], 30.0, atol=1e-3)


def test_a_direction_is_folded_into_a_line() -> None:
    ref = np.array([1.0, 0.0])
    vec = np.array([[math.cos(math.radians(170)), math.sin(math.radians(170))], [0.0, 0.0]])
    out = kk._signed_angle_2d(ref, vec)
    assert math.isclose(out[0], -10.0, abs_tol=1e-9)
    assert math.isnan(out[1])


def test_the_lateral_foot_speed_is_the_speed_across_the_kick() -> None:
    n = 20
    t = np.arange(n) / FPS
    foot = np.column_stack([10.0 * t, 8.4 * t, np.full(n, 0.1)])
    post = _posture({"left-big-toe-tip": foot}, n)
    assert np.allclose(post["lateral_foot_mps"][3:-3], 8.4, atol=1e-6)


def test_the_knee_share_is_one_when_the_knee_makes_all_of_the_foot_motion() -> None:
    n = 60
    t = np.arange(n) / FPS
    theta = 0.5 + 1.0 * t  # shank angle (rad), increasing at 1 rad/s
    hip = _static(n, (0, 0, 1))
    knee = _static(n, (0, 0, 0.5))
    ankle = knee + 0.5 * np.column_stack([np.sin(theta), np.zeros(n), -np.cos(theta)])
    foot = knee + 0.4 * np.column_stack(
        [np.sin(theta + 0.3), np.zeros(n), -np.cos(theta + 0.3)]
    )  # the foot is rigid with the shank: it turns about the knee with it
    knee_rate = kk.local_velocity(kk.angle_between(hip - knee, ankle - knee), FPS)
    share = kk._knee_share(hip, knee, ankle, foot, knee_rate, FPS)
    assert np.allclose(share[3:-3], 1.0, atol=0.02)


def test_the_knee_share_is_zero_when_the_leg_only_translates() -> None:
    n = 60
    t = np.arange(n) / FPS
    shift = np.column_stack([0.3 * t, np.zeros(n), np.zeros(n)])
    hip = _static(n, (0, 0, 1)) + shift
    knee = _static(n, (0, 0, 0.5)) + shift
    ankle = _static(n, (0.5, 0, 0.5)) + shift
    foot = _static(n, (0.5, 0.2, 0.5)) + shift
    knee_rate = kk.local_velocity(kk.angle_between(hip - knee, ankle - knee), FPS)
    share = kk._knee_share(hip, knee, ankle, foot, knee_rate, FPS)
    assert np.allclose(share[3:-3], 0.0, atol=1e-6)


def _ball() -> np.ndarray:
    f = np.arange(N)
    x = 10.0 + 30.0 * f / FPS
    return np.column_stack([x, np.full(N, 1.0), np.full(N, 1.72)])


def _pixels() -> dict[int, tuple[float, float]]:
    return {f: (500.0 + (50.0 if f >= CONTACT else 0.0), 500.0) for f in range(N)}


def _rows() -> list[dict]:
    ball = _ball()
    f = np.arange(N)
    foot = np.column_stack(
        [ball[CONTACT, 0] + 10.0 * (f - CONTACT) / FPS, np.full(N, 1.0), np.full(N, 1.72)]
    )
    support = np.tile([ball[CONTACT, 0], 0.7, 0.11], (N, 1))
    hip = np.column_stack([np.full(N, 5.0), 3.0 * f / FPS, np.full(N, 1.0)])
    knee = hip + np.array([0.0, 0.0, -0.5])
    ankle = knee + np.array([0.5, 0.0, 0.0])
    points = {
        "left-big-toe-tip": foot,
        "left-heel": foot,
        "right-ankle": support,
        "right-heel": support,
        "left-hip": hip,
        "right-hip": hip,
        "left-knee": knee,
        "left-ankle": ankle,
    }
    rows = []
    for name, arr in points.items():
        for frame in range(N):
            rows.append(
                {
                    "frame": frame,
                    "player_id": KICKER,
                    "kpt_idx": 0,
                    "kpt_name": name,
                    "x_m": float(arr[frame, 0]),
                    "y_m": float(arr[frame, 1]),
                    "z_m": float(arr[frame, 2]),
                    "valid": True,
                }
            )
    return rows


def test_the_new_metrics_are_reported_with_their_source_and_reason() -> None:
    result = kk.analyze_kicker(_rows(), _ball(), _pixels(), kicker_id=KICKER, fps=FPS)
    names = {m["metric"]: m for m in result["metrics"]}
    for name in (
        "pelvis_rotation_deg",
        "thigh_shank_plane_deg",
        "hip_angle_deg",
        "ankle_angle_deg",
        "lateral_foot_speed_mps",
        "knee_share_foot_speed",
    ):
        assert name in names
        assert names[name]["source"] in {"observado", "inferido", "filtrado"}
    # no shoulders in the scene: the hip angle is empty with the reason, never a number
    assert names["hip_angle_deg"]["value"] == ""
    assert "shoulder" in names["hip_angle_deg"]["reason"]
    assert result["context"]["contact"] == CONTACT


def test_a_speed_that_moves_with_the_window_is_flagged() -> None:
    rows = [{"metric": "v_foot_pre_mps", "value": 10.0, "quality": "ok", "reason": ""}]
    sensitivity = [
        {"window_s": 0.033, "v_foot_pre_mps": 10.0},
        {"window_s": 0.067, "v_foot_pre_mps": 20.0},
    ]
    kk._flag_window_sensitive(rows, sensitivity)
    assert rows[0]["quality"] == "window-sensitive"
    assert "100%" in rows[0]["reason"]


def test_write_outputs_writes_every_figure_in_both_languages(tmp_path: Path) -> None:
    result = kk.analyze_kicker(_rows(), _ball(), _pixels(), kicker_id=KICKER, fps=FPS)
    paths = kk.write_outputs(result, tmp_path / "kicker")
    for lang in ("pt", "en"):
        for key in ("sticks_side", "sticks_top", "curves"):
            assert paths[f"{key}_{lang}"].is_file() and paths[f"{key}_{lang}"].stat().st_size > 0
        html = paths[f"interactive_{lang}"]
        assert html.is_file() and "plotly" in html.read_text(encoding="utf-8")[:200000].lower()
    assert paths["figure"] == paths["curves_pt"]
    assert not any(key.startswith("overlay") for key in paths)  # no views, no overlay


def test_the_overlay_is_drawn_on_the_contact_frame_of_a_real_video(tmp_path: Path) -> None:
    centre, target = np.array([-40.0, -30.0, 15.0]), np.array([20.0, 0.0, 1.0])
    z = target - centre
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    dlt = compute_dlt3d_from_KRt(K, R, -R @ centre)
    video = tmp_path / "cam.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (320, 240))
    for _ in range(N):
        writer.write(np.full((240, 320, 3), 90, dtype=np.uint8))
    writer.release()
    view = CamView(
        name="ref",
        dlt=dict.fromkeys(range(N), dlt),
        markers={},
        cut_start=0,
        sync_cut=0,
        pts_cut=np.arange(N) / FPS,
    )
    result = kk.analyze_kicker(_rows(), _ball(), _pixels(), kicker_id=KICKER, fps=FPS)
    paths = kk.write_outputs(
        result, tmp_path / "kicker", views=[{"name": "ref", "view": view, "video": video}]
    )
    assert paths["overlay_ref_pt"].is_file() and paths["overlay_ref_en"].is_file()


def test_an_overlay_without_the_camera_frame_is_left_out(tmp_path: Path) -> None:
    view = CamView(
        name="ref", dlt={}, markers={}, cut_start=0, sync_cut=0, pts_cut=np.arange(N) / FPS
    )
    result = kk.analyze_kicker(_rows(), _ball(), _pixels(), kicker_id=KICKER, fps=FPS)
    paths = kk.write_outputs(
        result,
        tmp_path / "kicker",
        views=[{"name": "ref", "view": view, "video": tmp_path / "x.avi"}],
    )
    assert "overlay_ref_pt" not in paths


def test_a_missing_point_is_a_gap_not_a_zero() -> None:
    n = 4
    points = {"left-hip": _static(n, (0, 0, 1)), "left-knee": np.full((n, 3), np.nan)}
    segs = figs._segments(points, 0, (0, 2), "left", "right")
    assert segs["kick"] == []  # hip to knee needs both points
    assert figs._limits(points, np.full((n, 3), np.nan), range(n), (0, 2)) is not None
    assert figs._limits({}, np.full((n, 3), np.nan), range(n), (0, 2)) is None
