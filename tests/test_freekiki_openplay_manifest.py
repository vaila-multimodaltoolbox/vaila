"""Original-camera clock, identity, gaps and binocular reconstruction regression tests.

Version: 0.4.14
Update Date: 02 November 2026
"""

from __future__ import annotations

import json
from fractions import Fraction

import cv2
import numpy as np
import pytest

from vaila import freekiki_openplay_kinematics as kin
from vaila import freekiki_openplay_manifest as om
from vaila.freekiki_openplay_video import SCENES, frame_at_time, stable_speed, write_script


def test_registration_checks_original_pixels():
    rng = np.random.default_rng(2)
    seed = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    expected = np.array([[1.0, 0, 3], [0, 1, -2], [0, 0, 1]])
    original = cv2.warpPerspective(seed, expected, (320, 240))
    matrix, quality = om.register_original(seed, original)
    np.testing.assert_allclose(om.transform(matrix, [[150, 120]]), [[153, 118]], atol=0.15)
    assert quality["registration_inliers"] >= 12
    with pytest.raises(ValueError, match="features"):
        om.register_original(np.zeros_like(seed), original)


def test_smoothing_and_derivatives_never_bridge_a_gap():
    xy = np.column_stack((np.arange(25, dtype=float), np.zeros(25)))
    xy[12] = np.nan
    for window in (5, 7, 9):
        smooth = om.smooth_segments(xy, window)
        assert np.isnan(smooth[12]).all()
        assert kin.path_length(smooth) == pytest.approx(22)
        speed = kin.speed(smooth, 24)
        assert np.isnan(speed[12:14]).all()
        assert speed[15] == pytest.approx(24)
    assert kin.cod_events(np.array([0.0, np.nan, np.pi / 2])) == []


def test_ball_localization_rejects_yellowish_grass_and_keeps_gaps():
    hsv = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(hsv, (50, 50), 9, (25, 220, 220), -1)
    hsv[65:68, 55:58] = (38, 180, 180)  # nearer seed, but grass hue
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    point, area = om.ball_on_original(image, np.array([55, 66]))
    np.testing.assert_allclose(point, [50, 50], atol=0.1)
    assert area > 100
    assert om.ball_on_original(np.zeros_like(image), np.array([50, 50])) is None


def _manifest(tmp_path):
    dummy = tmp_path / "seed.csv"
    dummy.write_text("seed\n")
    video = tmp_path / "original.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 24, (64, 64))
    for _ in range(16):
        writer.write(np.full((64, 64, 3), 50, dtype=np.uint8))
    writer.release()
    cameras = []
    for cid, start, main in (("aberta", 4, False), ("principal", 1, True)):
        cameras.append(
            {
                "id": cid,
                "main": main,
                "range_1based": [start, start + 11],
                "video": str(video),
                **{
                    key: str(dummy)
                    for key in (
                        "seed_video",
                        "frame_map",
                        "dlt2d",
                        "dlt3d",
                        "ball_markers",
                        "field_points",
                        "seed_camera",
                        "seed_keypoints",
                    )
                },
            }
        )
    data = {
        "schema": "freekiki-dribble-tv-v1",
        "cameras": cameras,
        "participants": [
            {
                "id": 0,
                "name": "Driblador",
                "role": "dribbler",
                "color": "#ffe25c",
                "camera_ids": {"aberta": 41, "principal": 35},
            },
            {
                "id": 1,
                "name": "Defensor 1",
                "role": "defender",
                "color": "#5dd6ff",
                "camera_ids": {"aberta": 0, "principal": 23},
            },
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data))
    return path, data


def test_manifest_distinguishes_camera_ids_from_unique_people(tmp_path):
    path, data = _manifest(tmp_path)
    actual = om.load_manifest(path)
    assert actual["n_frames"] == 12
    assert actual["participants"][0]["camera_ids"] == {"aberta": 41, "principal": 35}
    data["participants"][1]["camera_ids"]["principal"] = 35
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="both dribbler and defender"):
        om.load_manifest(path)
    data["participants"][1]["camera_ids"]["principal"] = 23
    data["cameras"][0]["range_1based"][1] -= 1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="same number"):
        om.load_manifest(path)


def test_original_clock_offset_height_and_missing_pose(tmp_path):
    path, _ = _manifest(tmp_path)
    manifest = om.load_manifest(path)
    out = tmp_path / "results"
    # Cameras view a known nonzero-height ball from two different centres.
    matrices = [
        np.array([[40.0, 0, 0, 20], [0, 40, 0, 20], [0, 0, 0.1, 1]]),
        np.array([[40.0, 0, 10, 10], [0, 40, 0, 20], [0, 0, 0.1, 1]]),
    ]
    for ci, cam in enumerate(manifest["cameras"]):
        folder = out / "cameras" / cam["id"]
        rows3, rows2, ball, poses = [], [], [], []
        P = matrices[ci]
        H = P[:, [0, 1, 3]]
        for pair in range(12):
            f = pair + cam["range_1based"][0] - 1
            rows3.append(
                dict(
                    zip(
                        ["frame", *[f"L{i}" for i in range(1, 12)]],
                        [f, *P.ravel()[:11]],
                        strict=True,
                    )
                )
            )
            rows2.append(
                dict(
                    zip(
                        ["frame", *[f"L{i}" for i in range(1, 9)]], [f, *H.ravel()[:8]], strict=True
                    )
                )
            )
            uv = om.transform(P, [[pair / 24, 0.2, 0.8]])[0]
            ball.append({"frame": f, "p0_x": uv[0], "p0_y": uv[1]})
            for pid, delta in ((41 if ci == 0 else 35, 0), (0 if ci == 0 else 23, 2)):
                if pair == 6 and ci == 1 and pid == 23:
                    continue
                for name, y in (("left-ankle", -0.05), ("right-ankle", 0.05)):
                    uv = om.transform(H, [[pair / 24 + delta, y]])[0]
                    poses.append(
                        {
                            "frame": f,
                            "person_id": pid,
                            "kpt_name": name,
                            "x_px": uv[0],
                            "y_px": uv[1],
                        }
                    )
        om.write_csv(folder / "camera.dlt3d", rows3)
        om.write_csv(folder / "camera.dlt2d", rows2)
        om.write_csv(folder / "ball_markers.csv", ball)
        om.write_csv(folder / "sam3d" / "original_sam3dinov3_keypoints2d.csv", poses)
    analysis, _ = om.analyze_originals(manifest, out)
    assert analysis["quality"]["ball_accepted_pairs"] == 12
    np.testing.assert_allclose(np.array(analysis["ball_raw_xyz_m"])[:, 2], 0.8, atol=1e-7)
    assert analysis["dribbler"]["v_mean_mps"] == pytest.approx(1)
    assert analysis["markers"][0]["metrics"]["xy_m"][6] == [None, None]
    assert analysis["markers"][0]["metrics"]["speed_mps"][7] is None
    assert analysis["markers"][0]["vs_dribbler"]["min_distance_m"] == pytest.approx(2)
    provenance = om.read_csv(out / "data" / "ball_reprojection.csv")
    assert (provenance[0]["frame_open_1based"], provenance[0]["frame_main_1based"]) == ("4", "1")
    assert float(provenance[-1]["t_s"]) == pytest.approx(11 / 24)
    assert (out / "data" / "interactions_window_7.csv").is_file()


def test_editorial_clock_and_subtitles_are_separate(tmp_path):
    rate = Fraction(991001, 41333)
    assert frame_at_time(3, rate, 78, 0, 4, natural=True) == 71
    assert frame_at_time(10, rate, 78, 4, 16) == 39
    assert SCENES[0][0] == 0 and SCENES[-1][1] == 60
    assert all(a[1] == b[0] for a, b in zip(SCENES[:-1], SCENES[1:], strict=True))
    write_script(tmp_path)
    assert "00:01:00,000" in (tmp_path / "explicacao_tv.srt").read_text()
    data = {
        "dribbler": {"speed_mps": [5]},
        "filter_sensitivity": {
            str(w): {"dribbler": {"speed_mps": [v]}} for w, v in ((5, 9), (7, 5), (9, 2))
        },
    }
    assert stable_speed(data, 0) is None


def _turn_manifest_entries():
    return [
        {"id": "giro", "name": "Meia-volta", "primary": "body", "body_frames": [2, 10],
         "before": [0, 2], "after": [10, 14]},
        {"id": "curva", "name": "Curva", "primary": "path", "before": [20, 30],
         "after": [37, 47], "break_frame": 33},
    ]  # fmt: skip


def test_turns_are_validated_measured_and_split_the_tv_chapter():
    from vaila.freekiki_openplay_video import build_scenes, turn_summary, turn_text

    turns = _turn_manifest_entries()
    om.validate_turns(turns, 48)
    for bad, match in (
        ([{**turns[0], "primary": "x"}], "primary"),
        ([{**turns[1], "after": [25, 40]}], "before must end"),
        ([{**turns[1], "after": [37, 48]}], "0-based"),
        ([{k: v for k, v in turns[0].items() if k != "body_frames"}], "body_frames"),
        ([turns[1], turns[1]], "unique id"),
    ):
        with pytest.raises(ValueError, match=match):
            om.validate_turns(bad, 48)
    # An L path at 4 m/s (24 fps) and a hip line turning 180 degrees in both cameras.
    step = 4 / 24
    path = np.vstack(
        (
            np.column_stack((np.arange(32) * step, np.zeros(32))),
            np.column_stack((np.full(16, 31 * step), np.arange(1, 17) * step)),
        )
    )
    paths = dict.fromkeys(("5", "7", "9"), path)
    yaw = np.linspace(0.0, -180.0, 48)
    results = om.analyze_turns(turns, paths, {"aberta": yaw, "principal": yaw + 3}, 24.0)
    giro, curva = results
    assert giro["stable"] and giro["body"]["aberta"]["angle_deg"] == pytest.approx(
        -30.6383, abs=1e-3
    )
    assert curva["stable"] and curva["path"]["7"]["angle_deg"] == pytest.approx(90.0)
    summary = turn_summary(curva)
    assert summary["v_in_kmh"] == pytest.approx(14.4)
    head, detail = turn_text(summary)
    assert head.startswith("≈90° em") and "14 km/h" in detail
    scenes = build_scenes({"turns": results})
    assert scenes[0][0] == 0 and scenes[-1][1] == 60
    assert all(a[1] == b[0] for a, b in zip(scenes[:-1], scenes[1:], strict=True))
    assert [s[5] for s in scenes if len(s) > 5] == [0, 1]
    assert "Monteiro" in scenes[-1][4]
    assert build_scenes({}) == list(SCENES)
    unstable = dict(curva, stable=False)
    assert turn_text(turn_summary(unstable))[0] == "Ângulo: sem valor estável"
