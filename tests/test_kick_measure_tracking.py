"""Per-frame cameras of ``vaila.kick_measure``: files, frame-to-frame tracking, hold-out test.

The tracking tests render a synthetic broadcast: a random texture seen at infinity by a camera
that pans and zooms about a fixed centre, so every static feature obeys the pure-rotation law the
tracker relies on. The cameras of some frames are given (anchors); the others must be recovered
from the picture alone. No real video is needed.

Run: ``uv run --no-sync pytest tests/test_kick_measure_tracking.py -q``
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vaila import kick_measure as km

W, H = 640, 360
CENTRE = np.array([0.0, -75.0, 22.0])
N = 26
GAP = range(10, 16)  # frames without a known camera


def _camera(frame: int) -> np.ndarray:
    """Fixed centre, pan and zoom change every frame (a small version of a TV camera)."""
    target = np.array([-12.0 + 0.6 * frame, 0.0, 0.0])
    zc = (target - CENTRE) / np.linalg.norm(target - CENTRE)
    xc = np.cross(zc, [0.0, 0.0, 1.0])
    xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    R = np.array([xc, yc, zc])
    f = 1500.0 + 8.0 * frame
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    return K @ np.c_[R, -R @ CENTRE]


def _norm(P: np.ndarray) -> np.ndarray:
    return P / P[2, 3]


def _track(sources: dict[int, str] | None = None, n: int = N) -> km.CameraTrack:
    """A track with the given source per frame (default: dlt3d except the GAP, approximate)."""
    frames = {}
    for f in range(n):
        src = (sources or {}).get(f, km.SOURCE_APPROX if f in GAP else km.SOURCE_DLT3D)
        if src == km.SOURCE_APPROX:
            base = 9 if f < 13 else 16
            frames[f] = km.FrameCamera(
                f, _norm(_camera(base)), src, copied_from=base, focal_px=1500.0 + 8.0 * base
            )
        else:
            frames[f] = km.FrameCamera(
                f, _norm(_camera(f)), src, rms_px=0.5 if src == km.SOURCE_SOLVED else math.nan,
                n_points=5 if src == km.SOURCE_SOLVED else 0, focal_px=1500.0 + 8.0 * f,
            )  # fmt: skip
    return km.CameraTrack(frames, n, centre=CENTRE, principal_point=(W / 2, H / 2))


@pytest.fixture(scope="module")
def video(tmp_path_factory) -> Path:
    """Lossless-ish AVI of the synthetic broadcast (frame f is seen by ``_camera(f)``)."""
    import cv2

    rng = np.random.default_rng(3)
    tex = rng.integers(0, 256, (1800, 3000), dtype=np.uint8)
    tex = cv2.GaussianBlur(tex, (0, 0), 2.0)
    tex = cv2.normalize(tex, np.zeros_like(tex), 0, 255, cv2.NORM_MINMAX)
    off = np.array([[1, 0, 1200.0], [0, 1, 700.0], [0, 0, 1]])  # frame-0 pixel -> texture pixel
    P0 = _camera(0)
    _, R0 = km._decompose(P0)
    f0, _ = km._decompose(P0)
    K0 = np.array([[f0, 0, W / 2], [0, f0, H / 2], [0, 0, 1.0]])
    path = tmp_path_factory.mktemp("synthetic_video") / "synth.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (W, H))
    assert writer.isOpened(), "this OpenCV build cannot write MJPG"
    for f in range(N):
        fc, R = km._decompose(_camera(f))
        K = np.array([[fc, 0, W / 2], [0, fc, H / 2], [0, 0, 1.0]])
        M = K @ R @ R0.T @ np.linalg.inv(K0)  # frame-0 pixel -> frame-f pixel (pure rotation)
        frame = cv2.warpPerspective(tex, M @ np.linalg.inv(off), (W, H), flags=cv2.INTER_LINEAR)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    writer.release()
    return path


# ----------------------------------------------------------------------------------------------
# Ground homography and the per-frame files
# ----------------------------------------------------------------------------------------------


def test_ground_homography_is_the_z0_projection():
    P = _camera(3)
    H3 = km.ground_homography(P)
    assert H3[2, 2] == 1.0
    xy = np.array([[0.0, 0.0], [10.0, -4.0], [-25.0, 12.0], [3.5, 30.0]])
    uv = km.project_points(P, np.c_[xy, np.zeros(len(xy))])
    h = (H3 @ np.c_[xy, np.ones(len(xy))].T).T
    assert np.abs(h[:, :2] / h[:, 2:3] - uv).max() < 1e-6
    track = _track()
    assert np.allclose(track.homography(2), km.ground_homography(track.matrix(2)))
    with pytest.raises(ValueError):
        km.ground_homography(np.array([[1.0, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 0]]))


def test_write_read_frame_calibration_round_trip(tmp_path):
    sources = {0: km.SOURCE_SOLVED, 8: km.SOURCE_TRACKED, 9: km.SOURCE_TRACKED}
    track = _track(sources)
    written = km.write_frame_calibration(
        track, tmp_path / "sam" / "maincam", "maincam", video="/x/maincam.mp4", fps=30.0,
        width=W, height=H,
    )  # fmt: skip
    assert set(written) == {"dlt3d", "dlt2d", "cameras", "summary"}
    assert written["dlt3d"].name == "maincam_frames.dlt3d"
    assert written["dlt2d"].name == "maincam_frames.dlt2d"
    assert written["cameras"].name == "maincam_frames_cameras.csv"
    assert written["summary"].name == "maincam_frames_cameras.json"
    d3 = pd.read_csv(written["dlt3d"])
    d2 = pd.read_csv(written["dlt2d"])
    assert list(d3.columns) == ["frame"] + [f"L{i}" for i in range(1, 12)]
    assert list(d2.columns) == ["frame"] + [f"L{i}" for i in range(1, 9)]
    assert len(d3) == len(d2) == N and d3.frame.tolist() == list(range(N))
    assert np.isfinite(d3.to_numpy()).all() and np.isfinite(d2.to_numpy()).all()
    cams = pd.read_csv(written["cameras"])
    assert list(cams.columns) == ["frame", "source", "rms_px", "n_points", "focal_px", "from_frame"]
    assert cams.source[8] == "tracked" and cams.source[10] == "approximate"
    assert cams.from_frame[10] == 9 and pd.isna(cams.from_frame[3])
    summary = json.loads(written["summary"].read_text(encoding="utf-8"))
    assert summary["n_frames"] == N and (summary["width"], summary["height"]) == (W, H)
    assert summary["counts"] == {"dlt3d": 17, "solved": 1, "tracked": 2, "approximate": 6}
    assert summary["video"] == "/x/maincam.mp4" and summary["fps"] == 30.0
    assert summary["centre"] == pytest.approx(CENTRE.tolist())

    back = km.read_frame_calibration(written["dlt3d"])
    assert back.n_frames == N and back.principal_point == (W / 2, H / 2)
    for f in range(N):
        a, b = km.CameraTrack.matrix(track, f), back.matrix(f)
        assert np.abs(b - _norm(a)).max() <= 1e-9 * np.abs(b).max()
        assert back.source(f) == track.source(f)
    assert back.info(10).copied_from == 9 and back.info(10).from_frame == 9
    assert back.info(0).rms_px == pytest.approx(0.5) and back.info(0).n_points == 5
    assert back.counts() == track.counts()
    # the ground homography of the dlt3d is the dlt2d of the same frame (pixel error < 1e-6)
    grid = np.array([[x, y, 0.0] for x in (-20, 0, 15) for y in (-10, 0, 12)])
    for f in (0, 5, 12, N - 1):
        H3 = np.array([*d2.iloc[f, 1:].to_numpy(), 1.0]).reshape(3, 3)
        h = (H3 @ np.c_[grid[:, :2], np.ones(len(grid))].T).T
        assert np.abs(h[:, :2] / h[:, 2:3] - back.project(f, grid)).max() < 1e-6


def test_dlt2d_file_loads_in_quickmeasure(tmp_path):
    qm = pytest.importorskip("vaila.quickmeasure")
    written = km.write_frame_calibration(_track(), tmp_path, "maincam")
    cals = qm.QuickMeasureCalibration.from_dlt2d_sequence(str(written["dlt2d"]), total_frames=N)
    assert len(cals) == N
    # a pixel of the pitch measured with the calibration of that frame is the pitch point
    track = _track()
    for f in (0, 11, N - 1):
        uv = track.project(f, np.array([[4.0, -6.0, 0.0]]))[0]
        x, y = cals[f].pixel_to_real(uv[0], uv[1])
        assert (x, y) == pytest.approx((4.0, -6.0), abs=1e-6)


def test_write_marks_an_unusable_matrix_approximate_instead_of_hiding_it(tmp_path):
    track = _track()
    bad = np.array(track.matrix(4), dtype=float)
    bad[2, 3] = 0.0  # the pitch origin on the principal plane: no DLT layout
    track.frames[4] = km.FrameCamera(4, bad, km.SOURCE_SOLVED, rms_px=1.0, n_points=4)
    written = km.write_frame_calibration(track, tmp_path, "maincam")
    d3 = pd.read_csv(written["dlt3d"])
    assert np.isfinite(d3.to_numpy()).all() and len(d3) == N
    row = pd.read_csv(written["cameras"]).iloc[4]
    assert row.source == "approximate" and row.from_frame in (3, 5)


def test_find_frame_calibration_newest_and_stem_folder(tmp_path):
    work = tmp_path / "work" / "maincam"
    assert km.find_frame_calibration(work, "maincam") is None
    track = _track()
    old = work / "sam3dinov3" / "processed_sam3dinov3_20261101_000000" / "maincam"
    new = work / "sam3dinov3" / "processed_sam3dinov3_20261102_000000" / "maincam"
    p_old = km.write_frame_calibration(track, old, "maincam")["dlt3d"]
    p_new = km.write_frame_calibration(track, new, "maincam")["dlt3d"]
    import os

    os.utime(p_old, (1_000_000, 1_000_000))
    os.utime(p_new, (2_000_000, 2_000_000))
    assert km.find_frame_calibration(work, "maincam") == p_new
    assert km.find_frame_calibration(work.parent.parent, "maincam") == p_new  # the run folder
    assert km.find_frame_calibration(work / "sam3dinov3", "maincam") == p_new
    assert km.find_frame_calibration(new, "maincam") == p_new  # the stem folder itself
    assert km.find_frame_calibration(old.parent, "maincam") == p_old
    assert km.find_frame_calibration(work, "other_cam") is None
    assert km.sam_output_dir(work.parent.parent, "maincam") == new
    assert km.sam_output_dir(tmp_path / "nowhere", "maincam") is None


# ----------------------------------------------------------------------------------------------
# Tracking the approximate frames
# ----------------------------------------------------------------------------------------------


def _pitch_error(track: km.CameraTrack, f: int) -> float:
    grid = np.array([[x, y, 0.0] for x in np.linspace(-14, 14, 8) for y in np.linspace(-14, 14, 6)])
    a, b = km.project_points(_camera(f), grid), track.project(f, grid)
    ok = (a[:, 0] > 0) & (a[:, 0] < W) & (a[:, 1] > 0) & (a[:, 1] < H)
    assert ok.sum() >= 10
    return float(np.hypot(*(a[ok] - b[ok]).T).max())


def test_tracking_recovers_the_gap_from_the_picture_alone(video):
    track = _track()
    assert track.approximate_frames() == list(GAP)
    before = max(_pitch_error(track, f) for f in GAP)
    assert before > 10.0  # the copy of a neighbour stands still while the picture pans
    out = km.track_approximate_frames(track, video)
    assert out.approximate_frames() == []
    assert out.counts() == {"dlt3d": N - len(GAP), "tracked": len(GAP)}
    for f in GAP:
        info = out.info(f)
        assert info.source == km.SOURCE_TRACKED and info.from_frame in (9, 16)
        assert info.n_points >= km.TRACK_MIN_INLIERS and info.rms_px < 5.0
        assert _pitch_error(out, f) < 2.5, f"frame {f}"
    # frames that had a camera are untouched
    assert np.array_equal(out.matrix(3), track.matrix(3))
    # nearest anchor: the first half of the gap follows 9, the last half 16
    assert out.info(10).from_frame == 9 and out.info(15).from_frame == 16


def test_tracking_with_the_anchors_on_one_side_only(video):
    sources = dict.fromkeys(range(18, N), km.SOURCE_APPROX)  # the clip ends in an unknown stretch
    track = _track(sources)
    out = km.track_approximate_frames(track, video)
    tracked = [f for f in range(N) if out.source(f) == km.SOURCE_TRACKED]
    assert tracked, "a one-sided gap must still be tracked"
    assert out.info(18).from_frame == 17
    # drift grows with the distance: the first frames are the sharpest
    assert _pitch_error(out, 18) < 3.0


def test_tracking_a_featureless_video_keeps_approximate(tmp_path):
    import cv2

    path = tmp_path / "gray.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (W, H))
    for _ in range(N):
        writer.write(np.full((H, W, 3), 120, dtype=np.uint8))
    writer.release()
    track = _track()
    out = km.track_approximate_frames(track, path)
    assert out.approximate_frames() == list(GAP)  # nothing to follow: nothing is invented
    assert out.counts() == track.counts()


def test_track_features_masks_people_ball_and_overlays(video):
    boxes = {5: [(300.0, 150.0, 80.0, 120.0)]}
    ball = {5: (100.0, 250.0)}
    feat = km.track_features(video, [(4, 7)], boxes=boxes, ball_uv=ball)
    assert feat.frames() == [4, 5, 6, 7] and (feat.width, feat.height) == (W, H)
    _, uv = feat.obs[5]
    assert len(uv) > 200
    # no point inside the (grown) box, the ball disc or an overlay margin
    inside_box = (
        (uv[:, 0] > 300 - 8) & (uv[:, 0] < 380 + 8) & (uv[:, 1] > 150 - 12) & (uv[:, 1] < 270 + 12)
    )
    near_ball = np.hypot(uv[:, 0] - 100, uv[:, 1] - 250) < km.BALL_MASK_RADIUS_PX
    x0, y0, x1, y1 = km.OVERLAY_MARGINS[0]
    in_score = (uv[:, 0] < x1 * W) & (uv[:, 1] < y1 * H)
    assert not inside_box.any() and not near_ball.any() and not in_score.any()
    # the same feature id seen in two frames is one scene point: matches between 4 and 6
    a, b = feat.matches(4, 6)
    assert len(a) > 100 and np.median(np.hypot(*(b - a).T)) > 1.0
    assert len(feat.matches(4, 6, limit=30)[0]) == 30 and len(feat.matches(4, 40)[0]) == 0
    # tracks do not cross segments
    two = km.track_features(video, [(0, 1), (10, 11)])
    assert two.frames() == [0, 1, 10, 11]
    assert len(two.matches(1, 10)[0]) == 0


def test_validate_tracking_holdout_table(video):
    track = _track()
    template = np.array([[x, y, 0.0] for x in (-9.0, 0.0, 9.0) for y in (-6.0, 4.0, 10.0)])
    kps = {}
    for f in range(N):
        uv = km.project_points(_camera(f), template)
        kps[f] = {i: (float(u), float(v), 0.9) for i, (u, v) in enumerate(uv)
                  if 0 < u < W and 0 < v < H}  # fmt: skip
    feat = km.track_features(video, [(0, N - 1)])
    res = km.validate_tracking(track, feat, kps, template, ks=(3,), step=1, anchors_per_side=3)
    assert set(res) == {"one_sided", "two_sided"}
    for mode in ("one_sided", "two_sided"):
        s = res[mode][3]
        assert s["n_cases"] >= 2 and s["failed"] == 0
        assert len(s["cases"]) == s["n_cases"]
        assert s["field_px_median"] < 2.5 and s["ground_m_median"] < 0.3
        assert s["copy_px_median"] > 3 * s["field_px_median"]  # tracking beats copying
        assert s["floor_px_median"] < 0.01  # the reference is exact in this synthetic world
        assert not math.isnan(s["smooth_px_median"])
    text = km.format_validation(res)
    assert "one-sided" in text and "two-sided" in text and "   3 " in text
