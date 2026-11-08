"""Tests for vaila/chessboard_calib.py with synthetic cameras and boards (no real data).

Geometry: world Z up, three cameras 3-4 m in front of a 6x8 inner-corner board (0.1 m
squares) that rotates around a point 1 m above the floor. Truth is known exactly, so the
tests check recovery of corner order, intrinsics, frame offsets, camera poses and metric
board size through the exported DLTs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from vaila import chessboard_calib as cc

COLS, ROWS, SQ = 6, 8, 0.1
OBJ = cc.board_object_points(COLS, ROWS, SQ)
TARGET = np.array([0.0, 0.0, 1.0])


def look_at(center: np.ndarray, target: np.ndarray = TARGET) -> tuple[np.ndarray, np.ndarray]:
    z = (target - center) / np.linalg.norm(target - center)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    return R, -R @ center


def cameras(n: int = 3, radius: float = 3.5):
    out = []
    for k, az in enumerate(np.radians(np.linspace(-35, 35, n))):
        center = np.array([radius * math.sin(az), -radius * math.cos(az), 1.6 + 0.2 * k])
        out.append(look_at(center))
    return out


def board_pose(t: float) -> tuple[np.ndarray, np.ndarray]:
    """Upright board facing -Y (towards the cameras), rotating with an irregular speed."""
    R0 = np.column_stack([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])  # x_b=X, y_b=-Z, z_b=Y
    ax = math.radians(18) * math.sin(0.21 * t) + math.radians(6) * math.sin(0.57 * t + 1)
    az = math.radians(22) * math.sin(0.13 * t + 0.4) + math.radians(5) * math.sin(0.83 * t)
    R = cv2.Rodrigues(np.array([ax, 0.0, 0.0]))[0] @ cv2.Rodrigues(np.array([0.0, 0.0, az]))[0] @ R0
    centre_b = np.array([(COLS - 1) * SQ / 2, (ROWS - 1) * SQ / 2, 0.0])
    t_wb = (
        TARGET + np.array([0.05 * math.sin(0.1 * t), 0.0, 0.03 * math.cos(0.07 * t)]) - R @ centre_b
    )
    return R, t_wb


def project(K, dist, Rc, tc, Rb, tb):
    R = Rc @ Rb
    t = Rc @ tb + tc
    pts, _ = cv2.projectPoints(OBJ, cv2.Rodrigues(R)[0], t, K, dist)
    return pts.reshape(-1, 2), R, t


# --------------------------------------------------------------------------- #
# small units
# --------------------------------------------------------------------------- #


def test_parse_pattern_and_object_points():
    assert cc.parse_pattern("6x8") == (6, 8)
    assert cc.parse_pattern("6*8") == (6, 8)
    with pytest.raises(ValueError):
        cc.parse_pattern("6")
    assert OBJ.shape == (48, 3)
    assert np.allclose(OBJ[1] - OBJ[0], [SQ, 0, 0])
    assert np.allclose(OBJ[COLS] - OBJ[0], [0, SQ, 0])


def test_canonical_order_undoes_every_detector_ordering():
    K = np.array([[900.0, 0, 960], [0, 900.0, 540], [0, 0, 1]])
    Rc, tc = cameras()[1]
    Rb, tb = board_pose(0.0)
    truth, _, _ = project(K, np.zeros(5), Rc, tc, Rb, tb)
    g = truth.reshape(ROWS, COLS, 2)
    for cand in (g, g[::-1, ::-1], g[:, ::-1], g[::-1, :]):
        out = cc.canonical_corner_order(cand.reshape(-1, 2), COLS, ROWS)
        assert np.allclose(out, truth)


def _board_texture(px: int = 24, margin: int = 24) -> np.ndarray:
    h, w = (ROWS + 1) * px + 2 * margin, (COLS + 1) * px + 2 * margin
    tex = np.full((h, w), 255, np.uint8)
    for r in range(ROWS + 1):
        for c in range(COLS + 1):
            if (r + c) % 2 == 0:
                y0, x0 = margin + r * px, margin + c * px
                tex[y0 : y0 + px, x0 : x0 + px] = 0
    return tex


def render(K, Rc, tc, Rb, tb, size, px: int = 24, margin: int = 24) -> np.ndarray:
    """Pinhole rendering of the board texture (pixel centres at integer coordinates)."""
    R = Rc @ Rb
    t = Rc @ tb + tc
    H_img_board = K @ np.column_stack([R[:, 0], R[:, 1], t])
    s = SQ / px
    off = margin + px - 0.5  # texture coordinate of inner corner (0, 0)
    H_board_tex = np.array([[s, 0, -off * s], [0, s, -off * s], [0, 0, 1.0]])
    tex = _board_texture(px, margin)
    img = np.full((size[1], size[0]), 110, np.uint8)
    return cv2.warpPerspective(
        tex, H_img_board @ H_board_tex, size, dst=img, borderMode=cv2.BORDER_TRANSPARENT
    )


def test_detect_board_small_board_with_zoom_roi():
    K = np.array([[700.0, 0, 960], [0, 700.0, 540], [0, 0, 1]])
    Rc, tc = look_at(np.array([0.0, -6.5, 1.2]))
    Rb, tb = board_pose(1.0)
    img = render(K, Rc, tc, Rb, tb, (1920, 1080))
    truth, _, _ = project(K, np.zeros(5), Rc, tc, Rb, tb)
    width = np.ptp(truth[:, 0])
    assert width < 90  # a small board, as in the lab videos
    roi = cc._bbox(truth, 0.6, 1920, 1080)
    got = cc.detect_board(img, (COLS, ROWS), roi=roi, zoom=2.0, full_frame=False)
    assert got is not None
    err = np.linalg.norm(got - truth, axis=1)
    # ~10 px squares rendered without antialiasing: sub-pixel, not perfect
    assert err.mean() < 0.3 and err.max() < 0.5


def test_bundle_adjust_recovers_poses_noise_free():
    """Vectorized projection must match cv2.projectPoints: noise-free data -> ~0 px."""
    K = np.array([[1000.0, 0, 960], [0, 1000.0, 540], [0, 0, 1]])
    dist = np.array([-0.3, 0.08, 0.0, 0.0, 0.0])
    cams = cameras()
    intr = [{"K": K, "dist": dist} for _ in cams]
    obs, poses = [], [{} for _ in cams]
    for f in range(25):
        Rb, tb = board_pose(f)
        for c, (Rc, tc) in enumerate(cams):
            pts, R, t = project(K, dist, Rc, tc, Rb, tb)
            obs.append((c, f, pts))
            res = cc.board_pose(OBJ, pts, K, dist)
            assert res is not None and res[2] < 1e-3
            poses[c][f] = (res[0], res[1])
    cam_T0 = cc.initial_camera_poses(poses, ref=0)
    assert all(T is not None for T in cam_T0)
    cam_T0_ok = [T for T in cam_T0 if T is not None]
    board_T0 = {f: np.linalg.inv(cam_T0_ok[0]) @ cc._to_mat4(*poses[0][f]) for f in range(25)}
    cam_T, _board_T, info = cc.bundle_adjust(OBJ, obs, intr, cam_T0_ok, board_T0, ref=0)
    assert info["rms_px"] < 1e-3
    T0 = cc._to_mat4(*cams[0])
    for c in (1, 2):
        truth_rel = cc._to_mat4(*cams[c]) @ np.linalg.inv(T0)  # camera 0 -> camera c
        assert cc._rot_angle(cam_T[c][:3, :3] @ truth_rel[:3, :3].T) < math.radians(0.01)
        assert np.linalg.norm(cam_T[c][:3, 3] - truth_rel[:3, 3]) < 1e-4


def test_intrinsics_minimal_recovers_focal_and_k1():
    K = np.array([[1000.0, 0, 959.5], [0, 1000.0, 539.5], [0, 0, 1]])
    dist = np.array([-0.35, 0.0, 0.0, 0.0, 0.0])
    Rc, tc = cameras()[1]
    rng = np.random.default_rng(1)
    corners = {}
    for f in range(40):
        Rb, tb = board_pose(f)
        pts, _, _ = project(K, dist, Rc, tc, Rb, tb)
        corners[f] = pts + rng.normal(0, 0.1, pts.shape)
    out = cc.calibrate_intrinsics(OBJ, corners, (1920, 1080), mode="minimal")
    assert out["mode"] == "minimal"
    assert abs(out["K"][0, 0] - 1000.0) / 1000.0 < 0.02
    assert abs(out["dist"][0] - (-0.35)) < 0.08
    assert out["coverage"] < 0.05


def test_pairwise_lag_and_offsets():
    rng = np.random.default_rng(0)
    base = np.convolve(rng.normal(size=300), np.ones(7) / 7, mode="same")
    a = base.copy()
    b = np.roll(base, 4)  # event at frame f in a appears at f + 4 in b
    c = np.roll(base, -2)
    a[:20] = np.nan
    lag_ab = cc.pairwise_lag(a, b, 15)
    lag_ac = cc.pairwise_lag(a, c, 15)
    assert lag_ab is not None and abs(lag_ab[0] - 4) < 0.2
    assert lag_ac is not None and abs(lag_ac[0] + 2) < 0.2
    offsets, unlinked, exact = cc.solve_offsets(3, 0, {(0, 1): lag_ab, (0, 2): lag_ac})
    assert offsets == [0, 4, -2]
    assert unlinked == []
    assert exact[0] == 0.0 and abs(exact[1] - 4) < 0.2


def test_pairwise_lag_sub_frame():
    """Cameras offset by 2.4 frames: the lag is found between whole frames."""
    t = np.arange(400, dtype=float)

    def signal(x):
        return np.sin(0.11 * x) + 0.6 * np.sin(0.037 * x + 1.0) + 0.3 * np.sin(0.23 * x)

    a, b = signal(t), signal(t - 2.4)  # b shows at f + 2.4 what a shows at f
    lag = cc.pairwise_lag(a, b, 10)
    assert lag is not None and abs(lag[0] - 2.4) < 0.15


def test_umeyama_rigid_and_similarity():
    rng = np.random.default_rng(3)
    src = rng.normal(size=(6, 3))
    R = cv2.Rodrigues(np.array([0.2, -0.4, 0.9]))[0]
    t = np.array([1.0, -2.0, 0.5])
    s, R_est, t_est = cc.umeyama(src, (R @ src.T).T + t)
    assert s == 1.0 and np.allclose(R_est, R) and np.allclose(t_est, t)
    s2, R2, t2 = cc.umeyama(src, 1.7 * (R @ src.T).T + t, with_scale=True)
    assert np.isclose(s2, 1.7) and np.allclose(R2, R) and np.allclose(t2, t)


def test_intrinsics_toml_roundtrip_and_distort_tools_format(tmp_path):
    K = np.array([[1080.5, 0, 961.2], [0, 1079.0, 541.7], [0, 0, 1]])
    intr = {
        "K": K,
        "dist": np.array([-0.31, 0.05, 0.001, -0.002, 0.0]),
        "mode": "standard",
        "rms_px": 0.3,
        "coverage": 0.4,
    }
    path = tmp_path / "cam_intrinsics.toml"
    cc.write_intrinsics_toml(path, intr, (1920, 1080))
    back = cc.load_intrinsics_toml(path)
    assert np.allclose(back["K"], K) and np.allclose(back["dist"], intr["dist"])
    from vaila.vaila_lensdistortvideo import load_distortion_parameters

    params = load_distortion_parameters(str(path))
    assert set(params) >= {"fx", "fy", "cx", "cy", "k1", "k2", "k3", "p1", "p2"}
    assert np.isclose(params["k1"], -0.31)


def test_dlt3d_reprojects_like_pinhole(tmp_path):
    K = np.array([[1000.0, 0, 960], [0, 1000.0, 540], [0, 0, 1]])
    Rc, tc = cameras()[0]
    dlt = cc.write_dlt3d(tmp_path / "c.dlt3d", K, Rc, tc)
    X = np.array([0.3, -0.2, 1.4])
    L = dlt
    den = L[8] * X[0] + L[9] * X[1] + L[10] * X[2] + 1
    u = (L[0] * X[0] + L[1] * X[1] + L[2] * X[2] + L[3]) / den
    v = (L[4] * X[0] + L[5] * X[1] + L[6] * X[2] + L[7]) / den
    p = K @ (Rc @ X + tc)
    assert np.allclose([u, v], p[:2] / p[2], atol=1e-6)
    text = (tmp_path / "c.dlt3d").read_text().splitlines()
    assert text[0].startswith("frame,p1,") and len(text) == 2


# --------------------------------------------------------------------------- #
# end to end with rendered videos
# --------------------------------------------------------------------------- #


def _write_video(path: Path, frames: list[np.ndarray], fps: float = 30.0) -> None:
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"MJPG"), fps, (w, h))
    assert vw.isOpened()
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_GRAY2BGR))
    vw.release()


def test_end_to_end_synthetic_videos(tmp_path):
    size = (960, 540)
    K = np.array([[650.0, 0, 479.5], [0, 650.0, 269.5], [0, 0, 1]])
    cams = cameras(3, radius=3.0)
    n, late = 70, 3  # camera c3 starts 3 frames late
    vids = tmp_path / "videos"
    vids.mkdir()
    for c, (Rc, tc) in enumerate(cams):
        start = late if c == 2 else 0
        frames = [render(K, Rc, tc, *board_pose(0.6 * f), size) for f in range(start, n)]
        _write_video(vids / f"c{c + 1}.avi", frames)
    out = tmp_path / "out"
    rc = cc.main(
        [
            "-i",
            str(vids),
            "-o",
            str(out),
            "--workers",
            "1",
            "--max-lag",
            "10",
            "--holdout-block",
            "10",
            "--square",
            str(SQ),
        ]
    )
    assert rc == 0
    run = next(out.glob("processed_chessboard_calib_*"))
    for stem in ("c1", "c2", "c3"):
        for suffix in (".dlt3d", "_intrinsics.toml", "_board_pixels.csv"):
            assert (run / f"{stem}{suffix}").is_file()
        assert (run / "qa" / f"{stem}_coverage.png").is_file()
    report = json.loads((run / "calibration_report.json").read_text())
    off = report["sync"]["frame_offsets"]
    assert off["c3"] - off["c1"] == -late
    assert off["c2"] - off["c1"] == 0
    assert report["bundle_adjustment"]["rms_px"] < 0.5
    val = report["validation"]
    assert val["frames"] >= 20
    assert val["distance_rmse_median_m"] < 0.0015
    hold = val["holdout"]
    assert hold["test_frames"] >= 10 and hold["distance_rmse_median_m"] < 0.002
    for stem in ("c1", "c2", "c3"):
        f = report["cameras"][stem]["K"][0][0]
        assert abs(f - 650.0) / 650.0 < 0.03
    # world frame = upright board: the cameras sit in front (negative Y) and above it
    centres = np.array([report["cameras"][s]["camera_center_world_m"] for s in ("c1", "c2", "c3")])
    assert np.all(centres[:, 1] < 0)
    # the board turned, so no camera gets the static-board warning
    for stem in ("c1", "c2", "c3"):
        tilt = report["cameras"][stem]["intrinsics"]["board_tilt_p95_deg"]
        assert tilt > cc.STATIC_BOARD_TILT_DEG
    assert "kept almost one orientation" not in (run / "README_calibration.txt").read_text()

    # cut copies of the same videos load the lens of this run by camera name
    cut = tmp_path / "cut"
    cut.mkdir()
    for stem in ("c1", "c2", "c3"):
        (cut / f"{stem}_sync_5_frames_1_to_40.avi").symlink_to(vids / f"{stem}.avi")
    out2 = tmp_path / "out2"
    argv = ["-i", str(cut), "-o", str(out2), "--workers", "1", "--max-lag", "10"]
    argv += ["--holdout-block", "0", "--square", str(SQ), "--max-frames", "40"]
    argv += ["--frame-offsets", "0,0,-2.6"]  # decimals: kept for --subframe and RIFE
    assert cc.main([*argv, "--intrinsics-toml-dir", str(run)]) == 0
    report2 = json.loads(
        next(out2.glob("processed_chessboard_calib_*/calibration_report.json")).read_text()
    )
    for stem in ("c1", "c2", "c3"):
        cam = report2["cameras"][f"{stem}_sync_5_frames_1_to_40"]
        assert cam["intrinsics"]["mode"] == "loaded"
        assert cam["intrinsics"]["source"].endswith(f"{stem}_intrinsics.toml")
        assert np.allclose(cam["K"], report["cameras"][stem]["K"], atol=1e-4)
    sync2 = report2["sync"]
    c3 = "c3_sync_5_frames_1_to_40"
    assert sync2["mode"] == "manual"
    assert sync2["frame_offsets"][c3] == -3 and sync2["frame_offsets_subframe"][c3] == -2.6
    assert "--source-events" in sync2["rife_equalizer_command"]


def test_undistort_subcommand_matches_pinhole_and_applies_offsets(tmp_path):
    """Pixels undistorted by the subcommand are pinhole projections with the same K."""
    K = np.array([[1000.0, 0, 960], [0, 1000.0, 540], [0, 0, 1]])
    dist = np.array([-0.32, 0.09, 0.001, -0.0015, 0.0])
    Rc, tc = cameras()[1]
    calib = tmp_path / "processed_chessboard_calib_x"
    calib.mkdir()
    cams = {
        "cam_a": {
            "K": K.tolist(),
            "dist_k1_k2_p1_p2_k3": dist.tolist(),
            "frame_offset": 3,
            "fps": 30.0,
        },
        "cam_a_long": {"K": np.eye(3).tolist(), "dist_k1_k2_p1_p2_k3": [0] * 5, "frame_offset": 0},
    }
    (calib / "cameras.json").write_text(json.dumps(cams))
    pts3d = OBJ[:5] + np.array([0.0, 0.0, 1.0])
    ideal = (K @ (Rc @ pts3d.T + tc[:, None])).T
    ideal = ideal[:, :2] / ideal[:, 2:]
    dist_px, _ = cv2.projectPoints(pts3d, cv2.Rodrigues(Rc)[0], tc, K, dist)
    dist_px = dist_px.reshape(-1, 2)
    pix_dir = tmp_path / "pixels"
    pix_dir.mkdir()
    row: dict[str, float] = {"frame": 10}
    for k, (x, y) in enumerate(dist_px):
        row[f"p{k + 1}_x"], row[f"p{k + 1}_y"] = x, y
    row["p6_x"], row["p6_y"] = np.nan, np.nan  # a missing marker stays missing
    import pandas as pd

    pd.DataFrame([row]).to_csv(pix_dir / "cam_a_markers.csv", index=False)
    pd.DataFrame([row]).to_csv(pix_dir / "other_camera.csv", index=False)
    out = tmp_path / "und"
    rc = cc.main(
        ["undistort", "--calib", str(calib), "-i", str(pix_dir), "-o", str(out), "--apply-offsets"]
    )
    assert rc == 0
    assert not (out / "other_camera.csv").exists()  # no matching camera -> skipped
    got = pd.read_csv(out / "cam_a_markers.csv")
    assert int(got["frame"].iloc[0]) == 10 - 3
    for k in range(5):
        assert np.allclose(
            [got[f"p{k + 1}_x"].iloc[0], got[f"p{k + 1}_y"].iloc[0]], ideal[k], atol=0.02
        )
    assert np.isnan(got["p6_x"].iloc[0])
    assert cc._match_camera("cam_a_long_markers", ["cam_a", "cam_a_long"]) == "cam_a_long"


def test_detect_video_numbers_frames_by_timestamp_with_dropped_frames(tmp_path):
    """A camera that dropped frames: numbers follow the timestamps, not the read count."""
    import shutil
    import subprocess

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("needs ffmpeg/ffprobe")
    size = (960, 540)
    K = np.array([[650.0, 0, 479.5], [0, 650.0, 269.5], [0, 0, 1]])
    Rc, tc = cameras(3, radius=3.0)[1]
    n, dropped = 40, [12, 14, 15]
    truth = [project(K, np.zeros(5), Rc, tc, *board_pose(0.6 * f))[0] for f in range(n)]
    src = tmp_path / "cam.avi"
    _write_video(src, [render(K, Rc, tc, *board_pose(0.6 * f), size) for f in range(n)])
    dst = tmp_path / "cam.mp4"
    expr = "+".join(f"eq(n\\,{k})" for k in dropped)
    for codec in (["-c:v", "libx264", "-crf", "10"], ["-c:v", "mpeg4", "-q:v", "2"]):
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src)]
        cmd += ["-vf", f"select='not({expr})'", "-fps_mode", "passthrough", "-g", "10"]
        cmd += ["-bf", "0", *codec, "-pix_fmt", "yuv420p", str(dst)]
        if subprocess.run(cmd, capture_output=True).returncode == 0:
            break
    else:
        pytest.skip("ffmpeg could not encode the test video")
    det = cc.detect_video(dst, (COLS, ROWS), step=1)
    assert det.numbering == "timestamp"
    assert det.dropped == dropped
    assert det.fps == pytest.approx(30.0)
    assert set(det.corners) <= set(range(n)) - set(dropped)
    assert len(det.corners) >= n - len(dropped) - 2
    for f, c in det.corners.items():
        assert np.abs(c - truth[f]).mean() < 0.5, f
    # counting frames while reading would put every frame after the drops 3 frames early
    late = [f for f in det.corners if f > max(dropped)]
    shifted = np.mean([np.abs(det.corners[f] - truth[f - len(dropped)]).mean() for f in late])
    assert shifted > 1.0


# --------------------------------------------------------------------------- #
# lens from another calibration, static board, sub-frame offsets, RIFE command
# --------------------------------------------------------------------------- #


def test_lens_toml_found_for_cut_and_equalized_copies(tmp_path):
    for name in ("169.254.43.186_06_2024", "169.254.43.186_08_2024"):
        (tmp_path / f"{name}_intrinsics.toml").write_text("fx = 1\n")

    def find(video_name: str) -> str | None:
        path = cc.find_intrinsics_toml(tmp_path, video_name)
        return None if path is None else path.name

    assert find("169.254.43.186_06_2024") == "169.254.43.186_06_2024_intrinsics.toml"
    cut = "169.254.43.186_08_2024_sync_192_frames_15_to_195"
    assert find(cut) == "169.254.43.186_08_2024_intrinsics.toml"
    assert find(f"source_2_{cut}_equalized") == "169.254.43.186_08_2024_intrinsics.toml"
    assert find("169.254.43.186_07_2024") is None
    # the start of the name wins over a name found further inside it
    assert cc._match_camera("cam1_cam10_x", ["cam10", "cam1"]) == "cam1"
    assert cc._match_camera("source_cam10_x", ["cam1", "cam10"]) == "cam10"


def test_board_tilt_separates_a_static_board_from_a_moved_one():
    static = [cv2.Rodrigues(np.radians([2 * np.sin(k), 3 * np.cos(k), 1.0]))[0] for k in range(60)]
    moved = [
        cv2.Rodrigues(np.radians([40 * np.sin(k), 35 * np.cos(1.3 * k), 10.0]))[0]
        for k in range(60)
    ]
    assert cc.board_tilt_p95_deg(static) < cc.STATIC_BOARD_TILT_DEG
    assert cc.board_tilt_p95_deg(moved) > cc.STATIC_BOARD_TILT_DEG
    assert math.isnan(cc.board_tilt_p95_deg(static[:1]))


def _track(frames: np.ndarray, nan_at: tuple[int, ...] = ()):
    import pandas as pd

    df = pd.DataFrame({"frame": frames, "p1_x": 100 + 7.5 * frames, "p1_y": 50 - 2.0 * frames})
    df.loc[df["frame"].isin(nan_at), ["p1_x", "p1_y"]] = np.nan
    return df


@pytest.mark.parametrize("offset", [0.4, 0.561, -0.14, 4.681])
def test_subframe_shift_recovers_linear_motion(offset):
    out = cc.shift_to_reference(_track(np.arange(30)), offset)
    n = out["frame"].to_numpy()
    # reference frame n is camera position n + offset
    assert np.allclose(out["p1_x"], 100 + 7.5 * (n + offset))
    assert np.allclose(out["p1_y"], 50 - 2.0 * (n + offset))
    assert len(out) == 29  # one camera frame pair per reference frame
    assert out["frame"].dtype.kind == "i"


def test_subframe_shift_leaves_gaps_empty_and_integer_offset_only_renumbers():
    frames = np.array([k for k in range(30) if k != 12])  # a dropped frame
    out = cc.shift_to_reference(_track(frames, nan_at=(20,)), 0.5)
    n = set(out["frame"].tolist())
    assert {11, 12, 19, 20}.isdisjoint(n)  # each needs frame 12 or 20
    assert {10, 13, 18, 21} <= n
    same = cc.shift_to_reference(_track(np.arange(30)), 3.0)
    assert same["frame"].tolist() == list(range(-3, 27))
    assert np.allclose(same["p1_x"], 100 + 7.5 * np.arange(30))
    assert cc.shift_to_reference(_track(np.arange(30)), 2.9999999)["frame"].iloc[0] == -3


def test_undistort_subframe_interpolates_and_needs_offsets_before_rounding(tmp_path):
    import pandas as pd

    calib = tmp_path / "calib"
    calib.mkdir()
    K = [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]]
    cams = {"cam_b": {"K": K, "dist_k1_k2_p1_p2_k3": [0.0] * 5, "frame_offset": 1, "fps": 30.0}}
    (calib / "cameras.json").write_text(json.dumps(cams))
    pix = tmp_path / "pix"
    pix.mkdir()
    name = "cam_b_sync_192_frames_15_to_195_markers.csv"
    _track(np.arange(40)).to_csv(pix / name, index=False)
    argv = ["undistort", "--calib", str(calib), "-i", str(pix), "--subframe"]
    with pytest.raises(ValueError, match="before rounding"):
        cc.main([*argv, "-o", str(tmp_path / "a")])
    report = {"sync": {"frame_offsets_subframe": {"cam_b": 0.561}}}
    (calib / "calibration_report.json").write_text(json.dumps(report))
    assert cc.main([*argv, "-o", str(tmp_path / "b")]) == 0
    got = pd.read_csv(tmp_path / "b" / name)
    n = got["frame"].to_numpy()
    assert np.allclose(got["p1_x"], 100 + 7.5 * (n + 0.561), atol=1e-4)
    assert n[0] == 0 and len(got) == 39


def test_rife_command_keeps_every_position_inside_every_video(tmp_path):
    from vaila import video_event_equalizer as eq

    paths = ["c06.mp4", "c07.mp4", "c08.mp4"]
    cmd = cc.rife_alignment_command(paths, [181, 181, 181], [0.0, -0.14, 0.561], 0, tmp_path)
    assert cmd is not None
    flags = list(zip(cmd[4::2], cmd[5::2], strict=False))
    assert flags[:2] == [("--reference", "c06.mp4"), ("--reference-events", "2,2,180")]
    assert ("--source", "c07.mp4") in flags and ("--source-events", "1.860,1.860,179.860") in flags
    assert ("--source", "c08.mp4") in flags and ("--source-events", "2.561,2.561,180.561") in flags
    assert cmd[-2:] == ["--output-dir", str(tmp_path)]
    for value in ("2,2,180", "1.860,1.860,179.860", "2.561,2.561,180.561"):
        events = eq.parse_event_positions(value, "x")
        assert eq.check_events("x", events, 181) == "start"
    # whole-frame offsets are already handled by --apply-offsets: no command
    assert cc.rife_alignment_command(paths, [181] * 3, [0.0, 1.05, -2.0], 0, tmp_path) is None
